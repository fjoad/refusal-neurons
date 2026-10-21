"""Builds parallel English/Arabic splits for the Arabic teaser.

harmful:  MultiJail (Deng et al. 2024, MIT), 315 prompts human-translated into 9 languages.
          The same 100 prompts are sampled for English and Arabic, so the two sets are parallel.
harmless: Aya dataset test split (Singh et al. 2024, Apache-2.0), human-written prompts:
          100 Standard Arabic and 100 English (not parallel to each other; under 600 characters).
search:   128 more MultiJail prompts (en/ar, disjoint from the 100 above) and 128 Aya train
          prompts per language, for running neuron_search.py in each language.
"""
import csv
import io
import json
import random
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "splits"
SEED = 42
MULTIJAIL = "https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail/resolve/main/MultiJail.csv"
AYA_TEST = "https://huggingface.co/datasets/CohereLabs/aya_dataset/resolve/main/data/test-00000-of-00001.parquet"
AYA_TRAIN = "https://huggingface.co/datasets/CohereLabs/aya_dataset/resolve/main/data/train-00000-of-00001.parquet"


def fetch(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


def main():
    OUT.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    rows = list(csv.DictReader(io.StringIO(fetch(MULTIJAIL).decode("utf-8-sig"))))
    picked = rng.sample(rows, 100)
    splits = {"multijail_en": [r["en"] for r in picked], "multijail_ar": [r["ar"] for r in picked]}

    aya = pd.read_parquet(io.BytesIO(fetch(AYA_TEST)))
    aya = aya[aya.inputs.str.len() < 600]
    for lang, name in [("Standard Arabic", "aya_harmless_ar"), ("English", "aya_harmless_en")]:
        pool = aya[aya.language == lang].inputs.tolist()
        splits[name] = rng.sample(pool, 100)

    # Search training sets (128 each), disjoint from the teaser prompts above, so those stay held out.
    # Harmful: parallel MultiJail en/ar. Harmless: Aya *train* split, per language (not parallel).
    rest = [r for r in rows if r not in picked]
    train = rng.sample(rest, 128)
    splits["multijail_train_en"] = [r["en"] for r in train]
    splits["multijail_train_ar"] = [r["ar"] for r in train]
    aya_train = pd.read_parquet(io.BytesIO(fetch(AYA_TRAIN)))
    aya_train = aya_train[aya_train.inputs.str.len() < 600]
    for lang, name in [("Standard Arabic", "aya_train_ar"), ("English", "aya_train_en")]:
        pool = aya_train[aya_train.language == lang].inputs.tolist()
        splits[name] = rng.sample(pool, 128)

    for name, data in splits.items():
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
        print(f"{name}: {len(data)}")


if __name__ == "__main__":
    main()
