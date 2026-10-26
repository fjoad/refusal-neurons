"""Finds how a model starts its refusals in each language: the "refusal phrases" the search needs.

The search's push is the gradient of the probability that the answer starts with a refusal phrase
(Apple's Table 4 for English). In another language the model refuses in that language, so the
phrases must come from the model itself. For each harmful split, this generates a short unpinned
answer to every prompt and counts the most common openings (the first --prefix_tokens tokens,
which also works for Chinese, Japanese and Thai). Refusals are templated, so the most common
openings are refusal openings. The two most common, if each covers at least --min_share of the
answers, become that split's phrases. Only openings shared by at least --min_count answers are
printed or saved; full answers never are.
Usage:
  python replication/refusal_phrases.py --model Qwen/Qwen3-8B --splits poly_ar_harmful_train mj_bn_harmful_train
Then: neuron_search.py --phrases_json outputs/refusal_phrases/<model>_<tag>.json --phrases_key <split>
"""
import argparse
import collections
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_check import ROOT, build_prompts, load


@torch.no_grad()
def openings(model, tok, prompts, bs, max_new, prefix_tokens):
    out = []
    for i in range(0, len(prompts), bs):
        batch = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        ids = model.generate(**batch, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for row in ids[:, batch.input_ids.shape[1]:]:
            row = [t for t in row.tolist() if t not in tok.all_special_ids]
            out.append(tok.decode(row[:prefix_tokens], skip_special_tokens=True).strip())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--splits", nargs="+", required=True, help="harmful split names in data/splits/")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--max_new", type=int, default=16)
    ap.add_argument("--prefix_tokens", type=int, default=6)
    ap.add_argument("--min_count", type=int, default=3, help="only openings shared by this many answers are kept")
    ap.add_argument("--min_share", type=float, default=0.05, help="share of answers a phrase must cover")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left", local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto", local_files_only=True)
    model.eval()

    result = {"args": vars(args), "splits": {}}
    for split in args.splits:
        raw = load(split)[:args.n]
        prompts, _, _ = build_prompts(tok, raw)
        counts = collections.Counter(openings(model, tok, prompts, args.bs, args.max_new, args.prefix_tokens))
        top = [(o, c) for o, c in counts.most_common() if c >= args.min_count and o]
        phrases = [o for o, c in top[:2] if c / len(raw) >= args.min_share]
        result["splits"][split] = {"n": len(raw), "top": top[:10], "phrases": phrases}
        print(f"{split}: phrases {phrases}  (top: " + "; ".join(f"{o!r} x{c}" for o, c in top[:5]) +
              f")  ({(time.time() - t0) / 60:.1f} min)")

    out = ROOT / "outputs" / "refusal_phrases"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    name = f"{args.model.split('/')[-1]}{tag}.json"
    (out / name).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"saved outputs/refusal_phrases/{name}")


if __name__ == "__main__":
    main()
