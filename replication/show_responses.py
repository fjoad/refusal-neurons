"""Prints generated responses from an output JSON, for manual inspection by a human.

Kept separate from the experiment scripts so job logs contain metrics only. Reads neuron_check files (runs keyed
by pin value) and rerank / evaluate / direction_variants files (a list of runs with a pin or a mode).
Usage:
  python replication/show_responses.py outputs/neuron_check/<file>.json [--n 5] [--chars 300] [--out file.txt]
  # the same prompts side by side at chosen pins (for the pin-sweep example slide):
  python replication/show_responses.py outputs/rerank/Qwen3-8B_float16_pinsweep.json --pins 20,40,80 --by_prompt \
      --start 0 --n 5 --chars 400 --out pin_examples.txt
"""
import argparse
import json
from pathlib import Path


def run_list(d):
    """[(label, run)] for either file format."""
    if isinstance(d["runs"], dict):
        return [(f"pin={pin}", run) for pin, run in d["runs"].items()]
    out = []
    for run in d["runs"]:
        mode, pin = run.get("mode"), run.get("pin")
        where = f"L{run['layer']}:F{run['neuron']} " if run.get("layer") is not None and not mode else ""
        out.append((f"{where}{mode or ''}{' ' if mode and pin is not None else ''}{'pin=%g' % pin if pin is not None else ''}".strip() or "no pin", run))
    return out


def pin_of(label, run):
    pin = run.get("pin") if isinstance(run, dict) else None
    if pin is None and label.startswith("pin="):
        try:
            pin = float(label[4:])
        except ValueError:
            pin = None
    return pin


def verdicts(run, i):
    """The judges' labels for answer i, when the file has been judged."""
    keys = [k for k in ("unsafe", "unsafe_qwen3guard", "refusal_qwen3guard", "refusal_k1", "coherent_k1", "unsafe_k1")
            if isinstance(run.get(k), list) and len(run[k]) > i]
    return "  ".join(f"{k}={run[k][i]}" for k in keys)


def show(path, n=5, chars=300, pins=None, start=0, by_prompt=False):
    d = json.loads(Path(path).read_text())
    runs = run_list(d)
    if pins is not None:
        runs = [(label, run) for label, run in runs if (pin_of(label, run) is None and 0 in pins)
                or (pin_of(label, run) is not None and any(abs(abs(pin_of(label, run)) - abs(p)) < 1e-6 for p in pins))]
    lines = [f"# {Path(path).name}"]
    idx = range(start, min(start + n, len(d["prompts"])))
    if by_prompt:
        for i in idx:
            prompt = d["prompts"][i]
            lines.append(f"\n===== prompt {i}: {str(prompt)[:300]}")
            for label, run in runs:
                lines.append(f"\n[{label}]  {verdicts(run, i)}\n{run['responses'][i][:chars]}")
        return "\n".join(lines)
    for label, run in runs:
        lines.append(f"\n===== {label}  (string-match refusal rate {run['refusal_rate_stringmatch']:.2f})")
        for i in idx:
            lines.append(f"\n[PROMPT {i}] {str(d['prompts'][i])[:200]}\n[RESPONSE] {run['responses'][i][:chars]}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--n", type=int, default=5, help="how many prompts")
    ap.add_argument("--start", type=int, default=0, help="index of the first prompt")
    ap.add_argument("--chars", type=int, default=300)
    ap.add_argument("--pins", help="comma-separated pin sizes to keep (sign ignored; 0 = the unpinned run), e.g. 20,40,80")
    ap.add_argument("--by_prompt", action="store_true", help="group by prompt: one prompt, then its answer under each run")
    ap.add_argument("--out", help="write to this file instead of printing")
    args = ap.parse_args()
    pins = [float(p) for p in args.pins.split(",")] if args.pins else None
    text = show(args.path, args.n, args.chars, pins, args.start, args.by_prompt)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
