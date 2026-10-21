"""Counts how unpinned responses open, to find a model's refusal phrases in a language.

Reads a neuron_check output JSON and prints only the first few words of the
unpinned responses, and only openings shared by at least --min_count responses.
Refusals share stock openings; one-off openings (often compliance) are never printed.
Usage:
  python replication/refusal_openings.py outputs/neuron_check/<file>.json [--words 4] [--min_count 2]
"""
import argparse
import collections
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--words", type=int, default=4)
    ap.add_argument("--min_count", type=int, default=2)
    args = ap.parse_args()
    d = json.loads(Path(args.path).read_text())
    texts = d["runs"]["None"]["responses"]
    counts = {}
    for n in range(1, args.words + 1):
        counts[n] = collections.Counter(" ".join(t.split()[:n]) for t in texts if t.strip())
    print(f"{len(texts)} unpinned responses; openings shared by >= {args.min_count}:")
    for n in range(1, args.words + 1):
        common = [(o, c) for o, c in counts[n].most_common() if c >= args.min_count]
        print(f"  first {n} word(s): " + "; ".join(f"{o!r} x{c}" for o, c in common[:12]))


if __name__ == "__main__":
    main()
