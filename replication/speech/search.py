"""Apple's refusal-neuron search (Kazemi et al. 2026, Sec. 2) on the Qwen2.5-Omni thinker,
for typed or spoken prompts.

Port of Faaiz's replication/neuron_search.py: the hook, the score and the ranking are imported
from it unchanged (SearchHook, refusal_prefixes, top_candidates, watch_ranks). Only the forward
pass differs: it goes through the thinker so audio features and multimodal RoPE are handled.

score = (grad_harmful + grad_harmless) * (act_harmful - act_harmless) at the post-instruction
tokens, loss = -log(p_refusal / (1 - p_refusal)), first two-thirds of the layers; most negative first.

Usage: python search.py --modality text --phrases "I'm sorry, but I can't" ...
       python search.py --modality speech --voice v1 --phrases ... --watch 12:3456
"""
import argparse
import json
import time

import torch

from core import (OUT, SCORE_TOKENS, encode, forward_logits, load_model, speech_items, suffix_tokens,
                  text_items, text_layers)
from neuron_search import SearchHook, refusal_prefixes, top_candidates, watch_ranks  # Faaiz's code


def phrase_log_probability(model, batch, phrase):
    """log P(phrase | prompt) per prompt, teacher-forced; logits only at the phrase positions."""
    ids, mask = batch["input_ids"], batch["attention_mask"]
    target = torch.tensor(phrase, device=ids.device, dtype=torch.long)
    if len(phrase) > 1:
        cont = target[:-1].unsqueeze(0).expand(ids.shape[0], -1)
        ids = torch.cat((ids, cont), 1)
        mask = torch.cat((mask, torch.ones_like(cont)), 1)
    logits = forward_logits(model, batch, ids, mask, keep=len(phrase)).float()
    targets = target.view(1, -1, 1).expand(logits.shape[0], -1, -1)
    return logits.log_softmax(-1).gather(-1, targets).squeeze(-1).sum(-1)


def split_means(model, proc, hooks, items, prefixes, bs, label):
    """Same two-pass scheme as Faaiz's split_means (exact gradient of the log-odds loss)."""
    for hook in hooks:
        hook.reset()
    t0, expected = time.time(), 0
    for off in range(0, len(items), bs):
        batch = encode(proc, items[off:off + bs])
        if suffix_tokens(proc, batch) != SCORE_TOKENS or not torch.all(batch["attention_mask"][:, -len(SCORE_TOKENS):]):
            raise ValueError(f"unexpected template suffix {suffix_tokens(proc, batch)}")
        for hook in hooks:
            hook.prompt_len = batch["input_ids"].shape[1]
        log_probs = []
        with torch.no_grad():
            for j, phrase in enumerate(prefixes):
                for hook in hooks:
                    hook.record_acts = j == 0
                log_probs.append(phrase_log_probability(model, batch, phrase).double().cpu())
        for hook in hooks:
            hook.record_acts = False
        log_probs = torch.stack(log_probs)
        log_p = torch.logsumexp(log_probs, 0)
        if not torch.isfinite(log_probs).all() or not torch.all(log_p < 0):
            raise FloatingPointError("invalid refusal probability")
        weights = (log_probs - log_p).exp() / (-torch.expm1(log_p))
        for phrase, w in zip(prefixes, weights):
            lp = phrase_log_probability(model, batch, phrase)
            loss = -(lp * w.to(device=lp.device, dtype=lp.dtype)).sum()
            (loss * hooks[0].grad_scale).backward()
            del lp, loss
            expected += 1
        for hook in hooks:
            if hook.backward_calls != expected:
                raise RuntimeError("a hooked layer missed a gradient")
            if not torch.isfinite(hook.act_sum).all() or not torch.isfinite(hook.grad_sum).all():
                raise FloatingPointError(f"{label}: non-finite activation/gradient in batch at prompt {off} "
                                         "(padding? try --bs 1)")
        done = min(off + bs, len(items))
        if off == 0 or done % (8 * bs) == 0 or done == len(items):
            print(f"{label}: {done}/{len(items)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return ([h.act_sum / len(items) for h in hooks], [h.grad_sum / len(items) for h in hooks])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", choices=["text", "speech"], required=True)
    ap.add_argument("--voice", default=None, help="speech: which TTS voice")
    ap.add_argument("--harmful", default="arditi_harmful_train")
    ap.add_argument("--harmless", default="arditi_harmless_train")
    ap.add_argument("--phrases", nargs="+", required=True, help="refusal openings (this model's own)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_scale", type=float, default=128.0)
    ap.add_argument("--k", type=int, default=20, help="candidates kept (top 5 are the paper's)")
    ap.add_argument("--watch", nargs="*", default=[], metavar="L:F")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    if args.modality == "speech":
        harmful, harmless = speech_items(args.harmful, args.voice), speech_items(args.harmless, args.voice)
    else:
        harmful, harmless = text_items(args.harmful), text_items(args.harmless)
    model, proc = load_model()
    model.config.use_cache = False
    prefixes = refusal_prefixes(proc.tokenizer, args.phrases)
    layers = text_layers(model)
    n_layers = 2 * len(layers) // 3
    hooks, handles = [], []
    for layer in layers[:n_layers]:
        hook = SearchHook(len(SCORE_TOKENS), layer.mlp.down_proj.in_features, args.grad_scale)
        hooks.append(hook)
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    print(f"{args.modality} {args.voice or ''}: {len(harmful)} harmful / {len(harmless)} harmless; "
          f"layers 0..{n_layers - 1}; prefixes {[proc.tokenizer.decode(p) for p in prefixes]}", flush=True)
    try:
        harm = split_means(model, proc, hooks, harmful, prefixes, args.bs, "harmful")
        safe = split_means(model, proc, hooks, harmless, prefixes, args.bs, "harmless")
    finally:
        for h in handles:
            h.remove()
    out = OUT / "search"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.modality}{'_' + args.voice if args.voice else ''}{'_' + args.tag if args.tag else ''}"
    torch.save({"harmful": harm, "harmless": safe, "post_tokens": SCORE_TOKENS}, out / f"{name}_means.pt")
    cands = top_candidates(harm, safe, SCORE_TOKENS, k=args.k)
    watch = [tuple(int(x) for x in w.split(":")) for w in args.watch]
    watched = watch_ranks(harm, safe, SCORE_TOKENS, watch)
    for r, c in enumerate(cands[:10], 1):
        print(f"  {r:2d}. L{c['layer']}:F{c['neuron']}  score {c['score']:+.4g}  token {c['post_token']!r}  "
              f"harmful {c['mean_act_harmful']:+.3f} harmless {c['mean_act_harmless']:+.3f}")
    for w in watched:
        if w.get("searched"):
            print(f"  watch L{w['layer']}:F{w['neuron']}: rank {w['rank']}  harmful {w['mean_act_harmful']:+.3f} "
                  f"harmless {w['mean_act_harmless']:+.3f}  filter {'pass' if w['passes_filter'] else 'FAIL'}")
    (out / f"{name}.json").write_text(json.dumps({
        "args": vars(args), "n_harmful": len(harmful), "n_harmless": len(harmless), "n_layers_searched": n_layers,
        "refusal_prefixes": [proc.tokenizer.decode(p) for p in prefixes],
        "candidates": cands, "watched": watched, "minutes": (time.time() - t0) / 60}, indent=1, ensure_ascii=False))
    print(f"saved outputs/speech/search/{name}.json ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
