"""Compare two searches (e.g. text vs speech) over all neurons, not only their top 5.

Uses the per-layer means saved by search.py (*_means.pt) and scores every neuron exactly as
Faaiz's top_candidates does (best of the scored tokens, magnitude filter). Reports:
  - overlap of the top-k lists for several k
  - where each side's top 5 ranks in the other search
  - Spearman rank correlation of the scores over all neurons that pass both filters
Writes outputs/speech/search/compare_<a>_vs_<b>.json.

Usage: python compare_search.py text speech_v1
"""
import argparse
import json

import numpy as np
import torch

from core import OUT, REPO


def all_scores(name):
    """score[n] for every searched neuron (inf if it fails the magnitude filter), plus ids."""
    d = torch.load(OUT / "search" / f"{name}_means.pt")
    (ah, gh), (ash, gs) = d["harmful"], d["harmless"]
    scores, ids = [], []
    for layer, (a1, g1, a0, g0) in enumerate(zip(ah, gh, ash, gs)):
        s = (g1 + g0) * (a1 - a0)                      # [tokens, neurons]
        best, tok = s.min(0)
        n = torch.arange(s.shape[1])
        ok = a1[tok, n].abs() > a0[tok, n].abs()
        best = torch.where(ok, best, torch.full_like(best, float("inf")))
        scores.append(best.numpy())
        ids += [(layer, i) for i in range(s.shape[1])]
    return np.concatenate(scores), ids


def ranks(scores):
    order = np.argsort(scores, kind="stable")          # most negative first
    r = np.empty_like(order)
    r[order] = np.arange(1, len(order) + 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 50, 100, 500, 1000])
    args = ap.parse_args()
    sa, ids = all_scores(args.a)
    sb, ids_b = all_scores(args.b)
    assert ids == ids_b, "searches cover different neurons"
    ra, rb = ranks(sa), ranks(sb)
    name = lambda i: f"L{ids[i][0]}:F{ids[i][1]}"
    out = {"a": args.a, "b": args.b, "n_neurons": len(ids), "overlap": {}, "top5_a_in_b": [], "top5_b_in_a": []}
    print(f"{args.a} vs {args.b}: {len(ids):,} neurons searched")
    for k in args.ks:
        ta, tb = set(np.argsort(sa)[:k]), set(np.argsort(sb)[:k])
        out["overlap"][k] = len(ta & tb)
        print(f"  top {k:5d}: {len(ta & tb):4d} shared ({len(ta & tb) / k:.0%}); chance ≈ {k * k / len(ids):.2f}")
    for src, dst, sr, dr, key in ((args.a, args.b, ra, rb, "top5_a_in_b"), (args.b, args.a, rb, ra, "top5_b_in_a")):
        print(f"  {src} top 5 -> rank in {dst}:")
        for i in np.argsort(sr)[:5]:
            out[key].append({"neuron": name(i), "rank_src": int(sr[i]), "rank_dst": int(dr[i])})
            print(f"    {name(i):13s} {int(sr[i]):3d} -> {int(dr[i]):,}")
    both = np.isfinite(sa) & np.isfinite(sb)
    rho = float(np.corrcoef(ranks(sa[both]), ranks(sb[both]))[0, 1])
    out["spearman_all"] = rho
    out["n_both_pass_filter"] = int(both.sum())
    print(f"  Spearman rank correlation over {both.sum():,} neurons passing both filters: {rho:.3f}")
    path = OUT / "search" / f"compare_{args.a}_vs_{args.b}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"saved {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
