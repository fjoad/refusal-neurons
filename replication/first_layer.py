"""5 November follow-up, note 6, second half: at which layer do harmful and harmless prompts first separate, per language?

Hypothesis from the meeting: a small model may need an extra language-dependent step for rare languages, which
would show as the harmful/harmless signal appearing at a later layer there than in English (and later than in
the 8B model, as a share of depth).
For every language set (PolyRefuse 14, MultiJail 10): the first --n harmful and --n harmless test prompts, the
residual stream at the input of every layer (index = number of layers already applied; the last index is the
final layer's output) at Table 3's template tokens. Per layer and token: a difference-of-means direction fitted on half the prompts and
scored (AUROC) on the other half, both ways, averaged; the layer's score is its best token. Reported per set: the
first layer reaching 0.90 and 0.95, and the best layer ("own").
That first measure saturates after one layer (the two prompt groups differ in wording and topic, which a direction
fitted in the same language picks up at once), so the script also reports "English": the direction fitted on the
English prompts of the same dataset, scored on each other language. The datasets are parallel (the same prompts
translated), so this asks at which layer the harmful/harmless signal becomes one the languages share.
No generation; logs show numbers only.
Usage: python replication/first_layer.py --model Qwen/Qwen3-8B     (job: scripts/jobs/first_layer.sh)
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_check import ROOT, build_prompts, load, paper_score_tokens

LANGS = {"poly": "en ar de es fr it ja ko nl pl ru th yo zh".split(), "mj": "en ar zh it vi ko th bn sw jv".split()}


def auroc(pos, neg):
    """Chance that a value from pos is larger than one from neg (ties count half). pos, neg: [..., n]."""
    gt = (pos.unsqueeze(-1) > neg.unsqueeze(-2)).float().mean((-1, -2))
    eq = (pos.unsqueeze(-1) == neg.unsqueeze(-2)).float().mean((-1, -2))
    return gt + 0.5 * eq


@torch.no_grad()
def states(model, tok, prompts, n_post, n_paper, bs):
    """[prompts, layers + 1, n_paper tokens, d]: the residual stream at Table 3's tokens, entering each layer."""
    out = []
    for i in range(0, len(prompts), bs):
        batch = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        hs = model(**batch, output_hidden_states=True, logits_to_keep=1).hidden_states
        T = batch["input_ids"].shape[1]
        out.append(torch.stack([h[:, T - n_post:T - n_post + n_paper].float().cpu() for h in hs], dim=1))
    return torch.cat(out)


def separation(harm, safe):
    """[layers + 1]: held-out AUROC of a difference-of-means direction, best token per layer."""
    scores = 0
    for a, b in ((0, 1), (1, 0)):
        d = harm[a::2].mean(0) - safe[a::2].mean(0)                      # [layers, tokens, d]
        ph = (harm[b::2] * d).sum(-1).permute(1, 2, 0)                   # [layers, tokens, n]
        ps = (safe[b::2] * d).sum(-1).permute(1, 2, 0)
        scores = scores + auroc(ph, ps) / 2
    return scores.max(-1).values


def transfer(harm, safe, direction):
    """[layers + 1]: AUROC of another language's direction on these prompts, best token per layer; and the cosine
    between that direction and this language's own, averaged over tokens."""
    ph = (harm * direction).sum(-1).permute(1, 2, 0)
    ps = (safe * direction).sum(-1).permute(1, 2, 0)
    own = harm.mean(0) - safe.mean(0)
    return auroc(ph, ps).max(-1).values, torch.nn.functional.cosine_similarity(own, direction, dim=-1).mean(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left", local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto", local_files_only=True)
    model.eval()
    n_layers = len(model.model.layers)
    short = args.model.split("/")[-1]
    result = {"args": vars(args), "n_layers": n_layers, "sets": {}}
    print(f"{args.model}: {n_layers} layers; first layer whose input separates harmful from harmless (held-out AUROC of a difference-of-means direction)")
    print("set: own direction: first layer >= 0.90 / first layer >= 0.95 (as a share of depth) / best AUROC and its layer"
          " || English direction: the same three, then the cosine with the set's own direction at 25%, 50%, 75% of depth")
    for track, langs in LANGS.items():
        english = None  # the English direction of this dataset; English is first in each list
        for lang in langs:
            try:
                harmful, harmless = load(f"{track}_{lang}_harmful_test")[:args.n], load(f"{track}_{lang}_harmless_test")[:args.n]
            except FileNotFoundError:
                continue
            p_harm, n_post, post_tokens = build_prompts(tok, harmful)
            p_safe = build_prompts(tok, harmless)[0]
            n_paper = paper_score_tokens(args.model, post_tokens)
            harm, safe = states(model, tok, p_harm, n_post, n_paper, args.bs), states(model, tok, p_safe, n_post, n_paper, args.bs)
            sep = separation(harm, safe)
            first = lambda s, thr: next((l for l, x in enumerate(s.tolist()) if x >= thr), None)
            share = lambda l: "never" if l is None else f"{l} ({l / n_layers:.0%})"
            line = lambda s: f"{share(first(s, 0.90))} / {share(first(s, 0.95))} / {float(s.max()):.3f} at layer {int(s.argmax())}"
            entry = {"auroc_per_layer": sep.tolist(), "first_090": first(sep, 0.90), "first_095": first(sep, 0.95),
                     "best_layer": int(sep.argmax()), "n_harmful": len(harmful), "n_harmless": len(harmless)}
            text = f"  {track}_{lang} ({len(harmful)}/{len(harmless)}): own {line(sep)}"
            if english is None:
                english = harm.mean(0) - safe.mean(0)
            else:
                tr, cos = transfer(harm, safe, english)
                entry.update({"english_auroc_per_layer": tr.tolist(), "english_first_090": first(tr, 0.90),
                              "english_first_095": first(tr, 0.95), "cosine_with_english_per_layer": cos.tolist()})
                text += f" || English {line(tr)}; cosine " + " ".join(f"{float(cos[round(n_layers * f)]):+.2f}" for f in (0.25, 0.5, 0.75))
            print(text + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
            result["sets"][f"{track}_{lang}"] = entry
    out = ROOT / "outputs" / "first_layer"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{short}.json").write_text(json.dumps(result, indent=1))
    print(f"saved outputs/first_layer/{short}.json  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
