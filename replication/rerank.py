"""Milestone 2 rerank: sweep the pinned value m for each top-5 search candidate on HarmBench.

For each candidate neuron (from a neuron_search.py JSON), pin h_i <- m at every token
(prefill and generation), in the direction opposite to the neuron's harmful activation,
and generate on the HarmBench validation prompts. With --space residual, pin a decoder
input (resid_pre) dimension instead, sweeping BOTH +m and -m for every magnitude.
The space defaults to the search JSON's args.space, or mlp for older search files;
an explicit --space must agree with recorded metadata. One unpinned baseline is included.
Responses go to outputs/rerank/<model>_<dtype>_<tag>.json only (saved after every run);
residual output names add _residual, and neuron is the residual dimension index.
The log shows a rough string-match refusal rate. Then run replication/judge.py on the
output to get Llama-Guard ASR and pick the best (layer, neuron, m*).

The paper sweeps m over 0-20 (step 2) and 20-200 (step 5) with 512 new tokens; the default
here is a coarser grid and 128 new tokens, to fit our GPUs (see replication/README.md).

Usage: python replication/rerank.py --search outputs/neuron_search/Qwen3-8B_float16_arditi.json
       python replication/rerank.py --search outputs/neuron_search/Qwen3-8B_float16_residual.json --space residual
"""
import argparse
import json
import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_check import ROOT, build_prompts, generate, is_refusal, load


class PinHook:
    """Pre-down_proj or resid_pre hook that pins h[..., idx] to `value` (None = off)."""

    def __init__(self, idx):
        self.idx, self.value, self.records = idx, None, []  # records: unused, cleared by generate()

    def __call__(self, module, args, kwargs=None):
        if self.value is None:
            return None
        h = (args[0] if args else kwargs["hidden_states"]).clone()
        h[..., self.idx] = self.value
        if kwargs is None:
            return (h,) + args[1:]
        if args:
            return (h,) + args[1:], kwargs
        return args, dict(kwargs, hidden_states=h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", required=True, help="neuron_search.py output JSON (its top-5 candidates)")
    ap.add_argument("--space", choices=["mlp", "residual"], default=None,
                    help="feature space (default: search JSON's space, or mlp for older files)")
    ap.add_argument("--m", type=float, nargs="+", default=[2, 4, 6, 10, 15, 20, 30, 40, 60],
                    help="pin magnitudes; MLP opposes harmful activation, residual sweeps both signs")
    ap.add_argument("--split", default="harmbench_val")
    ap.add_argument("--n_prompts", type=int, default=0, help="0 = all")
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    search = json.loads((ROOT / args.search).read_text())
    search_space = search["args"].get("space")
    if search_space is not None:
        if search_space not in ("mlp", "residual"):
            ap.error(f"Unknown search space: {search_space}")
        if args.space is not None and args.space != search_space:
            ap.error(f"--space {args.space} disagrees with search space {search_space}")
    args.space = args.space or search_space or "mlp"
    model_name, candidates = search["args"]["model"], search["candidates"]
    prompts_raw = load(args.split)
    prompts_raw = prompts_raw[:args.n_prompts] if args.n_prompts else prompts_raw
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left", local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=getattr(torch, args.dtype), device_map="auto", local_files_only=True)
    model.eval()
    prompts, _, _ = build_prompts(tok, prompts_raw)

    out_dir = ROOT / "outputs" / "rerank"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    space = "_residual" if args.space == "residual" else ""
    out_path = out_dir / f"{model_name.split('/')[-1]}_{args.dtype}{tag}{space}.json"
    result = {"args": vars(args), "model": model_name, "prompts": prompts_raw, "runs": []}

    def run(layer, neuron, pin, m):
        hook = PinHook(neuron if neuron is not None else 0)
        decoder = model.model.layers[layer if layer is not None else 0]
        if args.space == "residual":
            handle = decoder.register_forward_pre_hook(hook, with_kwargs=True)
        else:
            handle = decoder.mlp.down_proj.register_forward_pre_hook(hook)
        hook.value = pin
        try:
            texts = generate(model, tok, hook, prompts, args.bs, args.max_new)
        finally:
            handle.remove()
        rr = sum(map(is_refusal, texts)) / len(texts)
        result["runs"].append({"layer": layer, "neuron": neuron, "m": m, "pin": pin,
                               "refusal_rate_stringmatch": rr, "responses": texts})
        out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        label = "baseline (no pin)" if pin is None else f"L{layer}:F{neuron} pin={pin:+g}"
        print(f"{label}: string-match refusal {rr:.2f}  ({(time.time() - t0) / 60:.1f} min)")

    n_pins = len(args.m) * (2 if args.space == "residual" else 1)
    print(f"{model_name}: {len(candidates)} candidates x {n_pins} pin values on {len(prompts)} {args.split} prompts")
    run(None, None, None, 0)
    for c in candidates:
        signs = [1.0, -1.0] if args.space == "residual" else [-math.copysign(1.0, c["mean_act_harmful"])]
        for m in args.m:
            for sign in signs:
                run(c["layer"], c["neuron"], sign * m, m)
    print(f"saved outputs/rerank/{out_path.name}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
