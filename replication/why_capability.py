"""Why does the constant pin cost ability, and is the anchor's zero cost a real result? (meeting notes 1 and 2)

On the MMLU questions, in one model load:
- how strongly the refusal neuron fires on each question (its strongest natural value over Table 3's tokens),
  how many questions pass a detector cut-off (--cutoff, raw activation), and the pin value the anchor (k = 1)
  would actually use on each question;
- accuracy, overall and split into "neuron fired" / "did not fire", under: no intervention; the neuron reset to
  its harmless mean; constant pins at fractions of m* (dose-response); the anchor; Arditi's all-layer ablation;
- how far each intervention moves the model's next-token distribution: mean KL(no intervention || intervention)
  at the answer position.
Hypotheses: (1) the constant pin costs ability because it puts the neuron far outside its natural range on every
prompt (then the cost grows with the pin and the reset costs nothing); (2) the neuron has other jobs (then even the
reset costs). The anchor's zero cost is uninformative if the neuron rarely fires on MMLU.
Prompts are exam questions; nothing is generated. Writes outputs/capability/<model>_L<l>_F<i>_<tag>.json.
Usage: python replication/why_capability.py --model Qwen/Qwen3-8B --layer 14 --neuron 7924 --pin 20 --cutoff -2.80 \
           --search outputs/neuron_search/Qwen3-8B_float16_arditi.json --arditi outputs/arditi/Qwen3-8B_jbb.pt
"""
import argparse
import contextlib
import json
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arditi import intervention
from capability import LETTERS, mmlu_text
from intervene import Pin
from neuron_check import ROOT, build_prompts, load, paper_score_tokens


def search_means(search_json, layer, neuron):
    d = json.loads((ROOT / search_json).read_text())
    for c in d["candidates"] + d.get("watched", []):
        if (c["layer"], c["neuron"]) == (layer, neuron) and "mean_act_harmful" in c:
            return c["mean_act_harmful"], c["mean_act_harmless"]
    raise ValueError(f"L{layer}:F{neuron} not in {search_json}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--neuron", type=int, required=True)
    ap.add_argument("--pin", type=float, required=True, help="m*, signed")
    ap.add_argument("--cutoff", type=float, required=True, help="detector cut-off in raw activation (XSTest optimal)")
    ap.add_argument("--search", required=True)
    ap.add_argument("--arditi", default=None, help="arditi.py .pt with a selected direction")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--mmlu", default="mmlu_1000")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--tag", default="why")
    args = ap.parse_args()

    t0 = time.time()
    a_harm, a_safe = search_means(args.search, args.layer, args.neuron)
    d = a_harm - a_safe
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left", local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto",
                                                 local_files_only=True)
    model.eval()
    pin = Pin(args.neuron)
    model.model.layers[args.layer].mlp.down_proj.register_forward_pre_hook(pin)
    direction = None
    if args.arditi:
        direction = torch.load(ROOT / args.arditi, map_location="cpu", weights_only=True).get("direction")

    data = load(args.mmlu)
    data = data[:args.n] if args.n else data
    prompts, _, post_tokens = build_prompts(tok, [mmlu_text(q) for q in data])
    n_post, n_paper = len(post_tokens), paper_score_tokens(tok.name_or_path, post_tokens)
    letter_ids = [tok(l, add_special_tokens=False).input_ids[0] for l in LETTERS]

    # mode -> pin value (None, float, or "anchor") and whether Arditi's ablation is on
    modes = [("none", None, False), ("reset to harmless mean", a_safe, False)]
    modes += [(f"constant {f * args.pin:+g}", f * args.pin, False) for f in args.fractions]
    modes += [("anchor k=1", "anchor", False)]
    if direction is not None:
        modes += [("direction ablated", None, True)]
    preds = {m: [] for m, _, _ in modes}
    kl = {m: [] for m, _, _ in modes}
    natural, anchor_pin = [], []

    for i in range(0, len(prompts), args.bs):
        batch = tok(prompts[i:i + args.bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        pin.values = None
        pin.records.clear()
        base = torch.log_softmax(model(**batch, logits_to_keep=1).logits[:, -1].float(), -1)
        acts = pin.records[0][:, -n_post:][:, :n_paper]  # left padding: the suffix is the last n_post tokens
        v = acts.max(1).values if d > 0 else acts.min(1).values
        h = args.pin * v / d - d  # anchor, k = 1 (paper Eq. 6-7)
        h = torch.minimum(h, torch.tensor(args.pin)) if args.pin > 0 else torch.maximum(h, torch.tensor(args.pin))
        natural += v.tolist()
        anchor_pin += h.tolist()
        for name, value, ablate in modes:
            pin.values = h.tolist() if value == "anchor" else value
            pin.records.clear()
            with intervention(model, direction) if ablate else contextlib.nullcontext():
                logp = torch.log_softmax(model(**batch, logits_to_keep=1).logits[:, -1].float(), -1)
            preds[name] += logp[:, letter_ids].argmax(-1).tolist()
            kl[name] += (base.exp() * (base - logp)).sum(-1).tolist()
        pin.values = None
        pin.records.clear()

    gold = [q["answer"] for q in data]
    fired = [(x <= args.cutoff) if d < 0 else (x >= args.cutoff) for x in natural]
    n, n_fired = len(data), sum(fired)
    print(f"{args.model} L{args.layer}:F{args.neuron}  m*={args.pin:+g}  harmful/harmless mean {a_harm:+.2f}/{a_safe:+.2f}  "
          f"cut-off {args.cutoff:+g}  on {n} {args.mmlu} questions")
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
    print(f"neuron's natural value: median {q(natural, .5):+.2f}, 5th-95th percentile {q(natural, .05):+.2f} to {q(natural, .95):+.2f}")
    print(f"questions where the neuron fires (passes the cut-off): {n_fired} of {n} ({n_fired / n:.1%})")
    by_subject = Counter(x.get("subject", "?") for x, f in zip(data, fired) if f).most_common(6)
    print("  subjects with the most firing questions: " + ", ".join(f"{s} {c}" for s, c in by_subject))
    capped = sum(abs(x - args.pin) < 1e-6 for x in anchor_pin)
    print(f"anchor pin actually used (k=1): median {q(anchor_pin, .5):+.2f}, 5th-95th percentile "
          f"{q(anchor_pin, .05):+.2f} to {q(anchor_pin, .95):+.2f}; at the full pin on {capped} questions")
    summary = {}
    for name, _, _ in modes:
        ok = [p == g for p, g in zip(preds[name], gold)]
        acc = sum(ok) / n
        acc_f = sum(o for o, f in zip(ok, fired) if f) / n_fired if n_fired else float("nan")
        acc_n = sum(o for o, f in zip(ok, fired) if not f) / (n - n_fired) if n - n_fired else float("nan")
        changed = sum(p != b for p, b in zip(preds[name], preds["none"])) / n
        mean_kl = sum(kl[name]) / n
        summary[name] = {"accuracy": acc, "accuracy_fired": acc_f, "accuracy_not_fired": acc_n,
                         "answers_changed": changed, "mean_kl": mean_kl}
        print(f"{name:>24}: accuracy {acc:.3f}  (fired {acc_f:.3f}, not fired {acc_n:.3f})  "
              f"answers changed {changed:.3f}  mean KL {mean_kl:.4f}")
    out = ROOT / "outputs" / "capability"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.model.split('/')[-1]}_L{args.layer}_F{args.neuron}_{args.tag}.json"
    path.write_text(json.dumps({"args": vars(args), "d": d, "mean_act_harmful": a_harm, "mean_act_harmless": a_safe,
                                "natural_value": natural, "anchor_pin": anchor_pin, "fired": fired, "gold": gold,
                                "preds": preds, "kl": kl, "summary": summary}, indent=1))
    print(f"saved {path.relative_to(ROOT)}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
