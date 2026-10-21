"""Per-token activations of one neuron across the whole chat-formatted prompt (paper Sections 3.1-3.2).

For each split, records the neuron's activation at every prompt token, then summarizes where
it fires: content tokens (the user's text) vs the post-instruction tokens (Table 3). This shows
the turn-boundary firing on harmful prompts (3.1), silence on benign prompts that contain
harmful words such as XSTest's homonyms (3.1 control), and whether a base checkpoint already
fires on the harmful content tokens instead (3.2). Prompts are inputs, so they are saved; no
generation happens.
Usage:
  python replication/token_acts.py --model Qwen/Qwen3-8B-Base --layer 14 --neuron 7924 \
      --splits jailbreakbench harmless_heldout xstest_safe --n 100 --tag base
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_check import ROOT, NeuronHook, build_prompts, load, paper_score_tokens


@torch.no_grad()
def token_acts(model, tok, hook, prompts):
    """One prompt at a time (no padding), so every token position is real."""
    out = []
    for p in prompts:
        ids = tok(p, return_tensors="pt", add_special_tokens=False).to(model.device)
        hook.records.clear()
        model(**ids)
        out.append((tok.convert_ids_to_tokens(ids.input_ids[0]), hook.records[0][0].tolist()))
    hook.records.clear()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--neuron", type=int, required=True)
    ap.add_argument("--splits", nargs="+", required=True, help="first split is treated as harmful for AUROCs")
    ap.add_argument("--n", type=int, default=100, help="prompts per split (0 = all)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto", local_files_only=True)
    model.eval()
    hook = NeuronHook(args.neuron)
    model.model.layers[args.layer].mlp.down_proj.register_forward_pre_hook(hook)

    result = {"args": vars(args), "splits": {}}
    summary = {}
    for split in args.splits:
        raw = load(split)
        raw = raw[:args.n] if args.n else raw
        prompts, n_post, post_tokens = build_prompts(tok, raw)
        n_paper = paper_score_tokens(args.model, post_tokens)
        rows = token_acts(model, tok, hook, prompts)
        # Content = the user's text: tokens before the template suffix, after the template prefix.
        n_prefix = len(tok(prompts[0].split(raw[0])[0], add_special_tokens=False).input_ids)
        content = [np.array(a[n_prefix:len(a) - n_post]) for _, a in rows]
        post = [np.array(a[len(a) - n_post:][:n_paper]) for _, a in rows]
        result["splits"][split] = {"prompts": raw, "post_tokens": post_tokens, "n_prefix": n_prefix,
                                   "tokens": [t for t, _ in rows], "acts": [a for _, a in rows]}
        summary[split] = {"content": content, "post": post}
        peak = lambda xs: float(np.mean([x[np.abs(x).argmax()] for x in xs if len(x)]))
        print(f"{split} ({len(rows)}): mean peak activation  content tokens {peak(content):+.3f}"
              f"   Table 3 tokens {peak(post):+.3f}")

    # Separation of the first split (harmful) from each other split, by where we read the neuron.
    first = args.splits[0]
    for other in args.splits[1:]:
        for where in ["content", "post"]:
            score = lambda xs: [float(np.abs(x).max()) if len(x) else 0.0 for x in xs]
            s_h, s_o = score(summary[first][where]), score(summary[other][where])
            auroc = roc_auc_score([1] * len(s_h) + [0] * len(s_o), s_h + s_o)
            result.setdefault("auroc", {})[f"{first}_vs_{other}_{where}"] = auroc
            print(f"AUROC {first} vs {other}, max |act| over {where} tokens: {auroc:.3f}")

    out = ROOT / "outputs" / "token_acts"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    name = f"{args.model.split('/')[-1]}_L{args.layer}_F{args.neuron}{tag}.json"
    (out / name).write_text(json.dumps(result, ensure_ascii=False))
    print(f"saved outputs/token_acts/{name}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
