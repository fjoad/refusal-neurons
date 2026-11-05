"""Milestone 3 capability cost: MMLU and GSM8K accuracy with and without the neuron intervention.

MMLU: next-token choice among A/B/C/D after the chat template (paper Appendix A).
GSM8K: greedy generation, then the number after '####' (or the last number) vs the gold answer.
Modes: none, constant (h_i <- m*), anchor (Eq. 7, per k), and with --arditi, Arditi's all-layer
ablation of a saved refusal direction. Same chat template as the attacks. --modes picks a subset.
--mmlu_instruction replaces the English answer instruction (e.g. Arabic for MMMLU).
Writes outputs/capability/<model>_L<l>_F<i>_<tag>.json and logs accuracies.
Usage:
  python replication/capability.py --model Qwen/Qwen3-8B --layer 14 --neuron 7924 --pin 20 \
      --search outputs/neuron_search/Qwen3-8B_float16_arditi.json
  Arditi's cost only: ... --modes arditi --arditi outputs/arditi/Qwen3-8B_jbb.pt --tag arditi
"""
import contextlib
import argparse
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arditi import intervention
from evaluate import generate
from intervene import Pin, anchor_values, neuron_gap
from neuron_check import ROOT, build_prompts, load

LETTERS = "ABCD"


MMLU_INSTRUCTION = "Answer with the letter of the correct option only."


def mmlu_text(q, instruction=MMLU_INSTRUCTION):
    options = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, q["choices"]))
    return f"{q['question']}\n\n{options}\n\n{instruction}"


def gsm_text(q):
    return f"{q['question']}\nSolve it step by step, then give the final answer on its own line as '#### <number>'."


def gsm_answer(text):
    m = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    nums = [m.group(1)] if m else re.findall(r"-?[\d,]*\.?\d+", text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


@torch.no_grad()
def mmlu_preds(model, tok, pin, prompts, values, bs):
    ids = [tok(l, add_special_tokens=False).input_ids[0] for l in LETTERS]
    preds = []
    for i in range(0, len(prompts), bs):
        batch = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        pin.values = values[i:i + bs] if isinstance(values, list) else values
        logits = model(**batch, logits_to_keep=1).logits[:, -1, ids]  # last position only: vocab is 128k-151k
        preds += logits.argmax(-1).tolist()
        pin.records.clear()
    pin.values = None
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--neuron", type=int, required=True)
    ap.add_argument("--pin", type=float, required=True, help="m*, signed")
    ap.add_argument("--search", required=True, help="neuron_search.py JSON that has this neuron (for d)")
    ap.add_argument("--k", type=float, nargs="*", default=[1, 2])
    ap.add_argument("--tasks", nargs="+", default=["mmlu", "gsm8k"])
    ap.add_argument("--mmlu", default="mmlu_1000")
    ap.add_argument("--gsm8k", default="gsm8k_250")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--n", type=int, default=0, help="first n items per task only (0 = all)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--tag", default="capability")
    ap.add_argument("--modes", nargs="+", default=["none", "constant", "anchor"],
                    choices=["none", "constant", "anchor", "arditi"])
    ap.add_argument("--arditi", help="arditi.py .pt with a selected direction; adds the 'arditi' mode")
    ap.add_argument("--mmlu_instruction", default=MMLU_INSTRUCTION)
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

    out_dir = ROOT / "outputs" / "capability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model.split('/')[-1]}_L{args.layer}_F{args.neuron}_{args.tag}.json"
    result = {"args": vars(args), "d": d, "results": []}
    modes = [(m, k) for m, ks in [("none", [None]), ("constant", [None]), ("anchor", args.k), ("arditi", [None])]
             if m in args.modes for k in ks]
    direction = None
    if args.arditi:
        direction = torch.load(ROOT / args.arditi, map_location="cpu", weights_only=True).get("direction")
        if direction is None:
            raise ValueError(f"{args.arditi} has no selected direction")
        modes += [] if "arditi" in args.modes else [("arditi", None)]
    elif "arditi" in args.modes:
        raise ValueError("--modes arditi needs --arditi")

    for task in args.tasks:
        data = load(getattr(args, task))
        data = data[:args.n] if args.n else data
        texts = [mmlu_text(q, args.mmlu_instruction) if task == "mmlu" else gsm_text(q) for q in data]
        prompts, _, post_tokens = build_prompts(tok, texts)
        for mode, k in modes:
            if mode in ("none", "arditi"):
                values = None
            elif mode == "constant":
                values = args.pin
            else:
                values = anchor_values(model, tok, pin, prompts, post_tokens, args.pin, d, k, args.bs)
            ctx = intervention(model, direction) if mode == "arditi" else contextlib.nullcontext()
            with ctx:
                if task == "mmlu":
                    preds = mmlu_preds(model, tok, pin, prompts, values, args.bs)
                else:
                    outs = generate(model, tok, pin, prompts, values, args.bs, args.max_new)
            if task == "mmlu":
                correct = [p == q["answer"] for p, q in zip(preds, data)]
                extra = {"preds": preds}
            else:
                answers = [gsm_answer(o) for o in outs]
                correct = [a is not None and abs(a - float(q["answer"])) < 1e-6 for a, q in zip(answers, data)]
                extra = {"answers": answers, "responses": outs}
            acc = sum(correct) / len(correct)
            result["results"].append({"task": task, "mode": mode, "k": k, "accuracy": acc, **extra})
            out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
            print(f"{task} {mode}{'' if k is None else f' k={k:g}'}: accuracy {acc:.3f}  ({(time.time() - t0) / 60:.1f} min)")
    print(f"saved outputs/capability/{out_path.name}")


if __name__ == "__main__":
    main()
