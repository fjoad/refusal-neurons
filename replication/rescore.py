"""Re-rank neurons from a saved search (`*_means.pt`) without rerunning the model.

Restricts the score to a subset of post-instruction tokens and reports the top
candidates plus the rank of any watched neurons, using the same score and
magnitude filter as neuron_search.py.

Usage:
  python replication/rescore.py outputs/neuron_search/Qwen3-8B_float16_t4_means.pt \
      --tokens 0 1 2 3 4 --watch 14:7924
"""
import argparse

import torch

from neuron_search import top_candidates, watch_ranks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("means", help="*_means.pt saved by neuron_search.py")
    ap.add_argument("--tokens", type=int, nargs="*", help="post-instruction token indices to keep (default: all)")
    ap.add_argument("--watch", nargs="*", default=[], metavar="L:F")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    saved = torch.load(args.means, weights_only=False)
    post_tokens = saved["post_tokens"]
    keep = args.tokens if args.tokens else list(range(len(post_tokens)))
    idx = torch.tensor(keep)
    pick = lambda split: ([t[idx] for t in split[0]], [t[idx] for t in split[1]])
    harmful, harmless = pick(saved["harmful"]), pick(saved["harmless"])
    tokens = [post_tokens[i] for i in keep]
    watch = [tuple(int(x) for x in w.split(":")) for w in args.watch]

    print(f"tokens kept: {list(zip(keep, tokens))}")
    for rank, c in enumerate(top_candidates(harmful, harmless, tokens, k=args.k), 1):
        print(f"  {rank}. L{c['layer']}:F{c['neuron']}  score={c['score']:+.4g}  token[{keep[c['post_token_index']]}]={c['post_token']!r}"
              f"  act={c['mean_act_harmful']:+.2f}/{c['mean_act_harmless']:+.2f}")
    for w in watch_ranks(harmful, harmless, tokens, watch):
        if not w["searched"]:
            print(f"  watch L{w['layer']}:F{w['neuron']}: layer not searched")
            continue
        print(f"  watch L{w['layer']}:F{w['neuron']}: rank {w['rank']}  score={w['score']:+.4g}"
              f"  token[{keep[w['post_token_index']]}]={w['post_token']!r}  filter={'pass' if w['passes_filter'] else 'FAIL'}")


if __name__ == "__main__":
    main()
