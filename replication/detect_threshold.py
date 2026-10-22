"""Turns a neuron's detection scores into a filter: the threshold, and how well it classifies.

Reads neuron_check.py outputs, which store one score per prompt (the neuron's most extreme value
over Table 3's tokens, sign-flipped so that higher = more harmful) and the true labels. Reports:
- AUROC (threshold-free)
- the threshold with the best accuracy on that same data (what Kazemi et al. call the "optimal
  threshold"), shown in raw activation units, with accuracy, precision and recall
- optionally (--calibrate), the threshold chosen on a different file applied to this one: a fair,
  deployable filter rather than an oracle one.
Usage: python replication/detect_threshold.py outputs/neuron_check/<xstest>.json [--calibrate <other>.json]
"""
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score


def load(path):
    d = json.load(open(path))
    return np.array(d["detection_scores"]), np.array(d["detection_labels"]), d["harmful_sign"]


def metrics(scores, labels, thr):
    pred = scores >= thr
    tp, fp = int((pred & (labels == 1)).sum()), int((pred & (labels == 0)).sum())
    acc = float((pred == (labels == 1)).mean())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / int((labels == 1).sum())
    return acc, prec, rec


def best_threshold(scores, labels):
    cands = np.unique(scores)
    accs = [metrics(scores, labels, t)[0] for t in cands]
    return float(cands[int(np.argmax(accs))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--calibrate", help="choose the threshold on this file, apply it to the others")
    args = ap.parse_args()
    cal = None
    if args.calibrate:
        s, l, sign = load(args.calibrate)
        cal = best_threshold(s, l)
        print(f"calibration {args.calibrate}: threshold {cal * sign:+.3f} (raw activation)")
    for path in args.paths:
        s, l, sign = load(path)
        thr = best_threshold(s, l)
        acc, prec, rec = metrics(s, l, thr)
        line = (f"{path}: AUROC {roc_auc_score(l, s):.3f} | optimal threshold: raw activation "
                f"{'<=' if sign < 0 else '>='} {thr * sign:+.3f} -> acc {acc:.3f} prec {prec:.3f} rec {rec:.3f}")
        if cal is not None:
            acc, prec, rec = metrics(s, l, cal)
            line += f" | calibrated threshold {cal * sign:+.3f} -> acc {acc:.3f} prec {prec:.3f} rec {rec:.3f}"
        print(line)


if __name__ == "__main__":
    main()
