"""Prints generated responses from a neuron_check output JSON, for manual inspection.

Kept separate from neuron_check.py so job logs contain metrics only.
Usage:
  python replication/show_responses.py outputs/neuron_check/<file>.json [--n 5] [--chars 300] [--out file.txt]
"""
import argparse
import json
from pathlib import Path


def show(path, n=5, chars=300):
    d = json.loads(Path(path).read_text())
    lines = [f"# {Path(path).name}"]
    for pin, run in d["runs"].items():
        lines.append(f"\n===== pin={pin}  (string-match refusal rate {run['refusal_rate_stringmatch']:.2f})")
        for prompt, resp in list(zip(d["prompts"], run["responses"]))[:n]:
            lines.append(f"\n[PROMPT] {prompt[:200]}\n[RESPONSE] {resp[:chars]}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--n", type=int, default=5, help="responses per pin value")
    ap.add_argument("--chars", type=int, default=300)
    ap.add_argument("--out", help="write to this file instead of printing")
    args = ap.parse_args()
    text = show(args.path, args.n, args.chars)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
