"""Milestone 3 attack evaluation: one neuron, constant and anchor interventions, on a test split.

Runs an unpinned baseline, the constant pin (h_i <- m*) and the anchor variant for each k,
on JailbreakBench by default with 512 new tokens as in the paper. Responses go to
outputs/evaluate/<model>_L<l>_F<i>_<tag>.json only; then run replication/judge.py on it for
Llama-Guard ASR. The log shows string-match refusal rates only.
Usage:
  python replication/evaluate.py --model Qwen/Qwen3-8B --layer 14 --neuron 7924 --pin 20 \
      --search outputs/neuron_search/Qwen3-8B_float16_arditi.json
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from intervene import Pin, anchor_values, neuron_gap
from neuron_check import ROOT, build_prompts, is_refusal, load


@torch.no_grad()
def generate(model, tok, pin, prompts, values, bs, max_new):
    """Greedy generation; values: None, a float, or one float per prompt."""
    texts = []
    for i in range(0, len(prompts), bs):
        batch = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        pin.values = values[i:i + bs] if isinstance(values, list) else values
        ids = model.generate(**batch, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        texts += tok.batch_decode(ids[:, batch.input_ids.shape[1]:], skip_special_tokens=True)
        pin.records.clear()
    pin.values = None
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--neuron", type=int, required=True)
    ap.add_argument("--pin", type=float, required=True, help="m*, signed (from the rerank)")
    ap.add_argument("--search", required=True, help="neuron_search.py JSON that has this neuron (for d)")
    ap.add_argument("--k", type=float, nargs="*", default=[1, 2], help="anchor scales; none = skip anchor")
    ap.add_argument("--split", default="jailbreakbench")
    ap.add_argument("--n", type=int, default=0, help="first n prompts only (0 = all)")
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--tag", default="jbb")
    args = ap.parse_args()

    t0 = time.time()
    d = neuron_gap(args.search, args.layer, args.neuron)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left", local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto", local_files_only=True)
    model.eval()
    pin = Pin(args.neuron)
    model.model.layers[args.layer].mlp.down_proj.register_forward_pre_hook(pin)
    raw = load(args.split)
    raw = raw[:args.n] if args.n else raw
    prompts, _, post_tokens = build_prompts(tok, raw)

    out_dir = ROOT / "outputs" / "evaluate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model.split('/')[-1]}_L{args.layer}_F{args.neuron}_{args.tag}.json"
    result = {"args": vars(args), "model": args.model, "d": d, "prompts": raw, "runs": []}
    print(f"{args.model} L{args.layer}:F{args.neuron}  m*={args.pin:+g}  d={d:+.3f}  on {len(raw)} {args.split} prompts")

    def run(mode, values, k=None):
        texts = generate(model, tok, pin, prompts, values, args.bs, args.max_new)
        rr = sum(map(is_refusal, texts)) / len(texts)
        result["runs"].append({"layer": None if mode == "none" else args.layer, "neuron": args.neuron,
                               "pin": None if mode == "none" else args.pin, "mode": mode, "k": k,
                               "refusal_rate_stringmatch": rr, "responses": texts})
        out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(f"{mode}{'' if k is None else f' k={k:g}'}: string-match refusal {rr:.2f}  ({(time.time() - t0) / 60:.1f} min)")

    run("none", None)
    run("constant", args.pin)
    for k in args.k:
        values = anchor_values(model, tok, pin, prompts, post_tokens, args.pin, d, k, args.bs)
        result.setdefault("anchor_values", {})[f"{k:g}"] = values
        run("anchor", values, k)
    print(f"saved outputs/evaluate/{out_path.name}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
