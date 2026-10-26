"""Builds the multilingual splits: PolyRefuse (14 languages) and MultiJail (10 languages).

PolyRefuse (Wang et al., NeurIPS 2025; github.com/mainlp/Multilingual-Refusal, Apache-2.0):
  Arditi's English splits machine-translated (Google Translate) into 13 languages, same order.
  poly_<lang>_harmful_train   128: the SAME 128 prompts as our English arditi_harmful_train,
                                   in every language (Arditi's random.Random(42) sample of 260)
  poly_<lang>_harmless_train  128: one fixed sample of their 200 harmless train prompts, same in every language
  poly_<lang>_harmful_test    572: their harmful test set
  poly_<lang>_harmless_test   500: their harmless test set (zh has none: its 72 unused harmless train prompts)
  poly_<lang>_harmful_rerank  191: from their harmful test set, for the rerank (like Apple's HarmBench-191)
  poly_<lang>_harmful_eval    100: disjoint from the rerank set, for the final attack (like JailbreakBench)
MultiJail (Deng et al. 2024, MIT): 315 harmful prompts human-translated into 9 languages.
  mj_<lang>_harmful_train     128: the same prompts as multijail_train_en/ar, in every language
  mj_<lang>_harmful_test      100: the same prompts as multijail_en/ar (the Arabic teaser set)
  mj_<lang>_harmful_rerank     87: the remaining MultiJail prompts, for the rerank
  mj_<lang>_harmless_train    128: Aya (Singh et al. 2024, Apache-2.0) prompts in that language, under 600 chars
  mj_<lang>_harmless_test     100: Aya test split where it has enough prompts, else unused Aya train prompts
                                   (Javanese has only 91 left). Aya prompts are not parallel across languages.
For English and Arabic the MultiJail and Aya picks equal the earlier multijail_*/aya_* splits.
Usage: python data/prepare_multilingual.py
"""
import csv
import io
import json
import random
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "splits"
POLY = "https://raw.githubusercontent.com/mainlp/Multilingual-Refusal/main"
POLY_LANGS = ["en", "ar", "de", "es", "fr", "it", "ja", "ko", "nl", "pl", "ru", "th", "yo", "zh"]
MULTIJAIL = "https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail/resolve/main/MultiJail.csv"
AYA = "https://huggingface.co/datasets/CohereLabs/aya_dataset/resolve/main/data/{}-00000-of-00001.parquet"
MJ_LANGS = {"en": "English", "ar": "Standard Arabic", "zh": "Simplified Chinese", "it": "Italian",
            "vi": "Vietnamese", "ko": "Korean", "th": "Thai", "bn": "Bengali", "sw": "Swahili", "jv": "Javanese"}


def fetch(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


def poly(path):
    return json.loads(fetch(f"{POLY}/{path}"))


def text(item):
    return item.get("instruction_translated") or item["instruction"]


def write(name, rows):
    assert rows and all(isinstance(r, str) and r for r in rows), name
    (OUT / f"{name}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))


def polyrefuse():
    # Same index draws as data/prepare_data.py (Arditi's pipeline: harmful sample first, from 260 prompts).
    harm_idx = random.Random(42).sample(range(260), 128)
    harmless_idx = random.Random(42).sample(range(200), 128)
    held_idx = random.Random(42).sample(range(572), 291)  # rerank (first 191) and final-test (last 100) prompts
    en_harm = [d["instruction"] for d in poly("dataset/splits/harmful_train.json")]
    assert [en_harm[i] for i in harm_idx] == json.loads((OUT / "arditi_harmful_train.json").read_text()), \
        "PolyRefuse English harmful train does not line up with arditi_harmful_train"
    en_harmless = [d["instruction"] for d in poly("dataset/splits/harmless_train_200_sampled.json")]
    for lang in POLY_LANGS:
        if lang == "en":
            harm, harmless = en_harm, en_harmless
        else:
            harm = [text(d) for d in poly(f"PolyRefuse/harmful_train_translated_{lang}.json")]
            harmless = [text(d) for d in poly(f"PolyRefuse/harmless_train_translated_{lang}.json")]
        assert len(harm) == 260 and len(harmless) == 200, lang
        write(f"poly_{lang}_harmful_train", [harm[i] for i in harm_idx])
        write(f"poly_{lang}_harmless_train", [harmless[i] for i in harmless_idx])
        harm_test = [text(d) for d in poly(f"PolyRefuse/harmful_test_translated_{lang}.json")]
        assert len(harm_test) == 572, lang
        write(f"poly_{lang}_harmful_test", harm_test)
        write(f"poly_{lang}_harmful_rerank", [harm_test[i] for i in held_idx[:191]])
        write(f"poly_{lang}_harmful_eval", [harm_test[i] for i in held_idx[191:]])
        try:
            harmless_test = [text(d) for d in poly(f"PolyRefuse/harmless_test_translated_{lang}.json")]
        except urllib.error.HTTPError:
            # PolyRefuse has no zh harmless test file: use the 72 harmless train prompts not in the search set.
            harmless_test = [harmless[i] for i in range(200) if i not in set(harmless_idx)]
            print(f"poly {lang}: no harmless test file; using {len(harmless_test)} unused harmless train prompts")
        write(f"poly_{lang}_harmless_test", harmless_test)
        print(f"poly {lang}: done")


def multijail():
    rows = list(csv.DictReader(io.StringIO(fetch(MULTIJAIL).decode("utf-8-sig"))))
    by_en = {r["en"]: r for r in rows}
    # Reuse the exact prompts already chosen for English/Arabic (data/prepare_arabic.py).
    test_rows = [by_en[e] for e in json.loads((OUT / "multijail_en.json").read_text())]
    train_rows = [by_en[e] for e in json.loads((OUT / "multijail_train_en.json").read_text())]
    used = {r["en"] for r in test_rows + train_rows}
    rerank_rows = [r for r in rows if r["en"] not in used]
    aya_train = pd.read_parquet(io.BytesIO(fetch(AYA.format("train"))))
    aya_test = pd.read_parquet(io.BytesIO(fetch(AYA.format("test"))))
    for lang, aya_name in MJ_LANGS.items():
        write(f"mj_{lang}_harmful_train", [r[lang] for r in train_rows])
        write(f"mj_{lang}_harmful_test", [r[lang] for r in test_rows])
        write(f"mj_{lang}_harmful_rerank", [r[lang] for r in rerank_rows])
        if lang in ("en", "ar"):  # identical to the earlier splits
            train = json.loads((OUT / f"aya_train_{lang}.json").read_text())
            test = json.loads((OUT / f"aya_harmless_{lang}.json").read_text())
        else:
            rng = random.Random(42)
            pool_tr = aya_train[(aya_train.language == aya_name) & (aya_train.inputs.str.len() < 600)].inputs.tolist()
            pool_te = aya_test[(aya_test.language == aya_name) & (aya_test.inputs.str.len() < 600)].inputs.tolist()
            train = rng.sample(pool_tr, 128)
            if len(pool_te) >= 100:
                test = rng.sample(pool_te, 100)
            else:
                rest = [p for p in pool_tr if p not in set(train)]
                test = rng.sample(rest, min(100, len(rest)))
        write(f"mj_{lang}_harmless_train", train)
        write(f"mj_{lang}_harmless_test", test)
        print(f"mj {lang}: harmful 128/100, harmless {len(train)}/{len(test)}")


if __name__ == "__main__":
    polyrefuse()
    multijail()
