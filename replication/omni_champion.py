"""One neuron for every language: scores that pick it from the searches already run (no GPU).

Every language's search saved each neuron's mean activation and gradient (*_means.pt), so each language's full
ranking of the 340,992 searched neurons can be rebuilt with the search's own rule. This script asks which single
neuron the languages agree on, in four ways:
  worst rank   a neuron's score is its WORST rank over the languages; the smallest wins. "Always in the top k."
  1/rank       the average of 1/rank over the languages (1.0 = first everywhere). Softer: one bad language
               does not decide everything.
  pooled       the saved means averaged over the languages and scored once: one search on all languages together.
  left out     for each language, the worst-rank winner chosen WITHOUT that language, and its rank in that language.
  shortlist vote  THE PICKER (28 November). In each list a neuron gets K points for 1st place, K-1 for 2nd, ...,
               1 point for K-th place and nothing below that (K = 5, the size of Kazemi et al.'s shortlist); its
               score is its points averaged over the lists and divided by K, so 1.0 = first in every list. One
               list can cost a neuron at most that list's own points however far down the neuron is there, so an
               outlying language cannot veto a neuron that is at the top everywhere else. Printed with it: in how
               many lists the winner is in the top K and WHICH lists it is not (the exceptions are named, not
               hidden); the winner for K = 3, 5, 10, 20; the winner with each list left out; and how often the
               same neuron wins when the lists are resampled with replacement (2,000 resamples, fixed seed).
Languages count when the model refuses at least --min_refusal of that language's harmful prompts with nothing
pinned (the rate the search saved); the others are listed and left out, and every table is also printed with all
languages. Scopes: PolyRefuse text, MultiJail text, both, PolyRefuse pictures, PolyRefuse text and pictures.
Usage: sbatch scripts/cpu.sh replication/omni_champion.py
Results log: '2026-11-28: One neuron for every language, chosen by a score'.
"""
import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "neuron_search"
POLY = "en ar de es fr it ja ko nl pl ru th yo zh".split()
MJ = "en ar zh it vi ko th bn sw jv".split()
KS = [1, 2, 3, 5, 10, 20, 50, 100]


def load_set(stem):
    """(search JSON, means [layers, 4, tokens, width]: harmful act, harmful grad, harmless act, harmless grad)."""
    js, pt = OUT / f"{stem}.json", OUT / f"{stem}_means.pt"
    if not js.exists() or not pt.exists():
        return None
    d = json.loads(js.read_text())
    m = torch.load(pt, map_location="cpu", weights_only=True)
    n = d["args"].get("score_tokens") or len(m["post_tokens"])
    (ah, gh), (as_, gs) = m["harmful"], m["harmless"]
    means = torch.stack([torch.stack([a[:n].float(), g[:n].float(), b[:n].float(), h[:n].float()])
                         for a, g, b, h in zip(ah, gh, as_, gs)])
    return d, means


def rank_of(means):
    """Rank of every neuron (flat index layer * width + neuron; 1 = best) by the search's rule: best token per
    neuron, magnitude filter at that token, global sort by score. Neurons failing the filter share the last rank."""
    ah, gh, as_, gs = means[:, 0], means[:, 1], means[:, 2], means[:, 3]
    score, tok = ((gh + gs) * (ah - as_)).min(dim=1)  # [layers, width]
    valid = ah.gather(1, tok.unsqueeze(1)).squeeze(1).abs() > as_.gather(1, tok.unsqueeze(1)).squeeze(1).abs()
    flat = score.flatten().clone()
    flat[~valid.flatten()] = float("inf")
    order = torch.argsort(flat, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(1, len(order) + 1)
    rank[~valid.flatten()] = int(valid.sum()) + 1
    return rank


def points(R, k):
    """Shortlist-vote score of every neuron: mean over lists of max(0, k + 1 - rank) / k."""
    return (k + 1 - R).clamp(min=0).mean(0) / k


def vote(R, tags, name, k=5, n_boot=2000):
    sc = points(R, k)
    order = torch.argsort(-sc)[:5]
    w = int(order[0])
    print(f"SHORTLIST VOTE (K = {k}): " + "  ".join(f"{name(i)} {sc[i]:.3f}" for i in order))
    inside = R[:, w] <= k
    print(f"  winner {name(w)}: in the top {k} of {int(inside.sum())}/{len(tags)} lists; median rank {int(R[:, w].median())}; "
          f"not in the top {k} in: " + (", ".join(f"{t} (rank {int(R[j, w])})" for j, t in enumerate(tags) if not inside[j]) or "none"))
    print("  winner for other K: " + "  ".join(f"K={kk}: {name(int(torch.argmax(points(R, kk))))} {points(R, kk).max():.3f}" for kk in (3, 5, 10, 20)))
    print(f"  runner-up {name(order[1])}: in the top {k} of {int((R[:, order[1]] <= k).sum())}/{len(tags)} lists; margin {sc[order[0]] - sc[order[1]]:.3f}")
    cand = torch.nonzero((R <= 20).any(0)).squeeze(1)  # only neurons in some list's top 20 can score
    P = (k + 1 - R[:, cand]).clamp(min=0) / k
    loo = [int(cand[torch.argmax(torch.cat([P[:j], P[j + 1:]]).mean(0))]) for j in range(len(tags))]
    print(f"  one list left out: the same winner in {sum(x == w for x in loo)}/{len(tags)} folds"
          + "".join(f"; without {tags[j]}: {name(x)}" for j, x in enumerate(loo) if x != w))
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(len(tags), (n_boot, len(tags)), generator=g)
    wins = torch.stack([P[i].mean(0) for i in idx]).argmax(1)
    top = torch.bincount(wins, minlength=len(cand))
    best = torch.argsort(-top)[:3]
    print(f"  lists resampled {n_boot} times: " + "  ".join(f"{name(cand[i])} wins {100 * int(top[i]) / n_boot:.1f}%" for i in best if top[i] > 0))


def analyse(title, sets, min_refusal, width):
    name = lambda i: f"L{int(i) // width}:F{int(i) % width}"
    for label, tags in (("languages that count", [t for t in sets if sets[t]["counts"]]), ("all languages", list(sets))):
        if label == "all languages" and len(tags) == sum(sets[t]["counts"] for t in sets):
            continue
        if len(tags) < 2:
            continue
        left = [t for t in sets if t not in tags]
        print(f"\n===== {title}: {label} ({len(tags)} lists" + (f"; left out, refusal below {min_refusal:g}: {' '.join(left)}" if left else "") + ")")
        R = torch.stack([sets[t]["rank"] for t in tags]).float()  # [lists, neurons]
        worst, where = R.max(0)
        mrr = (1 / R).mean(0)
        in5 = (R <= 5).sum(0)
        print("in every list's top k:  " + "  ".join(f"k={k}: {int((worst <= k).sum())}" for k in KS))
        print("by worst rank (ties: higher average of 1/rank):")
        for i in torch.argsort(worst - 0.5 * mrr)[:8]:
            print(f"  {name(i):12} worst rank {int(worst[i]):6d} (in {tags[int(where[i])]})  second worst {int(R[:, i].sort().values[-2]):6d}  "
                  f"mean 1/rank {mrr[i]:.3f}  in top 5 of {int(in5[i]):2d}/{len(tags)}  first in {int((R[:, i] == 1).sum())}")
        print("by average of 1/rank:")
        for i in torch.argsort(-mrr)[:5]:
            print(f"  {name(i):12} mean 1/rank {mrr[i]:.3f}  worst rank {int(worst[i]):6d} (in {tags[int(where[i])]})  in top 5 of {int(in5[i]):2d}/{len(tags)}")
        pooled = rank_of(torch.stack([sets[t]["means"] for t in tags]).mean(0))
        top = torch.argsort(pooled)[:5]
        print("pooled search (means averaged over the lists): top 5  " + "  ".join(name(i) for i in top))
        champ = int(torch.argsort(worst - 0.5 * mrr)[0])
        print(f"  the worst-rank winner {name(champ)} is number {int(pooled[champ])} in the pooled search")
        print(f"  ranks of {name(champ)} per list: " + "  ".join(f"{t} {int(R[j, champ])}" for j, t in enumerate(tags)))
        vote(R, tags, name)
        if label == "languages that count" and len(tags) > 2:
            print("one list left out: winner chosen from the others, and its rank in the list left out")
            same = 0
            for j, t in enumerate(tags):
                keep = [x for x in range(len(tags)) if x != j]
                w, m = R[keep].max(0).values, (1 / R[keep]).mean(0)
                c = int(torch.argsort(w - 0.5 * m)[0])
                same += c == champ
                print(f"  without {t:14} winner {name(c):12} (worst rank elsewhere {int(w[c])})  its rank in {t}: {int(R[j, c])}")
            print(f"  the same winner in {same} of {len(tags)} folds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen2.5-Omni-7B")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--min_refusal", type=float, default=0.5)
    args = ap.parse_args()
    data, width = {}, None
    for form in ("text", "image"):
        for track, langs in (("poly", POLY), ("mj", MJ)):
            for l in langs:
                got = load_set(f"{args.model}_{args.dtype}_omni_{form}_{track}_{l}")
                if got is None:
                    continue
                d, means = got
                width = means.shape[-1]
                rank = rank_of(means)
                saved = [c["layer"] * width + c["neuron"] for c in d["candidates"]]
                if torch.argsort(rank)[:len(saved)].tolist() != saved:
                    print(f"WARNING {form} {track}_{l}: rebuilt top {len(saved)} differs from the saved one")
                rr = d.get(f"refusal_rate_{form}", d.get("refusal_rate_text"))
                data[(form, track, l)] = {"rank": rank, "means": means, "refusal": rr,
                                          "counts": rr is None or rr >= args.min_refusal}
    print("unpinned refusal rate saved by each search (text / pictures):")
    for track, langs in (("poly", POLY), ("mj", MJ)):
        print(f"  {track}: " + "  ".join(f"{l} {data[('text', track, l)]['refusal'] if ('text', track, l) in data else '—'}"
                                         f"/{data[('image', track, l)]['refusal'] if ('image', track, l) in data else '—'}" for l in langs))
    pick = lambda forms, tracks: {f"{f[0]}:{t}_{l}": v for (f, t, l), v in data.items() if f in forms and t in tracks}
    scopes = [("PolyRefuse, text", pick(["text"], ["poly"])), ("MultiJail, text", pick(["text"], ["mj"])),
              ("Both datasets, text", pick(["text"], ["poly", "mj"])), ("PolyRefuse, pictures", pick(["image"], ["poly"])),
              ("PolyRefuse, text and pictures", pick(["text", "image"], ["poly"]))]
    for title, sets in scopes:
        if sets:
            analyse(title, sets, args.min_refusal, width)


if __name__ == "__main__":
    main()
