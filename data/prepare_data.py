"""Builds the Kazemi et al. (2026) data splits from Arditi et al.'s processed files.

train:  128 harmful (AdvBench, MaliciousInstruct, TDC2023) + 128 harmless (Alpaca)
val:    HarmBench standard behaviors (val + test = 200) minus JailbreakBench overlaps (paper: 191)
test:   JailbreakBench (100)
heldout_harmless: 100 Alpaca prompts disjoint from train, used for activation plots
"""
import json
import random
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset"
OUT = Path(__file__).parent / "splits"
SEED = 42


def fetch(path):
    with urllib.request.urlopen(f"{BASE}/{path}") as r:
        return json.load(r)


def instructions(rows):
    return [r["instruction"] for r in rows]


def main():
    OUT.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    jbb = instructions(fetch("processed/jailbreakbench.json"))
    harmbench = instructions(fetch("processed/harmbench_val.json")) + instructions(fetch("processed/harmbench_test.json"))
    jbb_set = {x.strip().lower() for x in jbb}
    harmbench_val = [x for x in harmbench if x.strip().lower() not in jbb_set]

    harmful_pool = []
    for name in ["advbench", "malicious_instruct", "tdc2023"]:
        harmful_pool += instructions(fetch(f"processed/{name}.json"))
    eval_set = jbb_set | {x.strip().lower() for x in harmbench}
    harmful_pool = [x for x in dict.fromkeys(harmful_pool) if x.strip().lower() not in eval_set]

    harmless_train_pool = instructions(fetch("splits/harmless_train.json"))
    harmless_test_pool = instructions(fetch("splits/harmless_test.json"))

    splits = {
        "harmful_train": rng.sample(harmful_pool, 128),
        "harmless_train": rng.sample(harmless_train_pool, 128),
        "harmbench_val": harmbench_val,
        "jailbreakbench": jbb,
        "harmless_heldout": rng.sample(harmless_test_pool, 100),
    }
    for name, rows in splits.items():
        (OUT / f"{name}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(f"{name}: {len(rows)}")
    print(f"harmbench before JBB filter: {len(harmbench)} (paper: 200 -> 191)")


if __name__ == "__main__":
    main()
