"""Picks a model's refusal tokens in a language for Arditi's refusal score (arditi.py).

Arditi scores refusal by the probability of a few first-response tokens ("I", "As" in English).
Here they are chosen from data: the first token of each unpinned answer in judged rerank files,
split by Qwen3Guard's refusal label. Tokens that open refusals far more often than other answers
are kept, greedily, until they cover --coverage of the refusals. The log shows token strings and
counts only, never answers. Writes outputs/refusal_tokens/<model>_<tag>.json, read by
arditi.py --refusal_tokens_json.
Usage:
  python replication/refusal_tokens.py --model Qwen/Qwen3-8B --tag ar \
      outputs/rerank/Qwen3-8B_float16_poly_ar.json outputs/rerank/Qwen3-8B_float16_mj_ar.json
"""
import argparse
import collections
import json

from transformers import AutoTokenizer

from neuron_check import ROOT


def unpinned_run(d):
    runs = d["runs"].values() if isinstance(d["runs"], dict) else d["runs"]
    for run in runs:
        if run.get("pin") in (None, "None") and run.get("layer") is None:
            return run
    raise ValueError("no unpinned run in this file")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="judged rerank/evaluate JSONs with refusal_qwen3guard")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="ar")
    ap.add_argument("--precision", type=float, default=0.8, help="min share of a token's answers that are refusals")
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--coverage", type=float, default=0.8, help="stop once this share of refusals is covered")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    counts = collections.defaultdict(lambda: [0, 0])  # token id -> [refusals, others]
    n_refusal = 0
    for path in args.paths:
        run = unpinned_run(json.loads((ROOT / path).read_text()))
        for text, refused in zip(run["responses"], run["refusal_qwen3guard"]):
            ids = tok.encode(text, add_special_tokens=False)
            if not ids:
                continue
            counts[ids[0]][0 if refused else 1] += 1
            n_refusal += refused
    rows = sorted(counts.items(), key=lambda kv: -kv[1][0])
    print(f"{n_refusal} refusals among {sum(a + b for a, b in counts.values())} unpinned answers")
    print("first token: refusals / other answers")
    for i, (r, o) in rows[:15]:
        print(f"  {i:7d} {tok.decode([i])!r:>12}  {r:4d} / {o:4d}")

    chosen, covered = [], 0
    for i, (r, o) in rows:
        if covered >= args.coverage * n_refusal:
            break
        if r + o >= args.min_count and r / (r + o) >= args.precision:
            chosen.append(i)
            covered += r
    if not chosen:
        raise ValueError("no token passes the precision and count thresholds")
    print(f"chosen: {chosen} = {[tok.decode([i]) for i in chosen]}, covering {covered}/{n_refusal} refusals")
    out = ROOT / "outputs" / "refusal_tokens" / f"{args.model.rstrip('/').split('/')[-1]}_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "token_ids": chosen,
                               "tokens": [tok.decode([i]) for i in chosen],
                               "covered": covered, "n_refusal": n_refusal,
                               "counts": {str(i): c for i, c in rows}}, indent=1, ensure_ascii=False))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
