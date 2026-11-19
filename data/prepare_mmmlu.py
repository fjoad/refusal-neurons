"""MMLU in the project's other languages: the same 1000 questions as mmlu_1000, translated.

Two sources, both translations of the full MMLU test set, so every language gets the same 1000 questions:
- OpenAI's MMMLU (professional translations; test/mmlu_<XX-YY>.csv, row by row in the order of cais/mmlu 'all'):
  ar, bn, de, es, fr, it, ja, ko, sw, yo, zh.
- Cohere's Global-MMLU (sample_id = <subject>/test/<position within the subject>), for project languages that
  MMMLU lacks: nl, pl, ru, vi.
Thai and Javanese are in neither; no split is written for them (17 November meeting, point 30: a language with
no MMLU translation cannot be checked this way).
Each mmlu_1000 question is found in the English set by its subject and text, then the same row is taken from
the translation. The translation's own answer letter is kept, since it goes with the translation's own options.
It differs from the English key on a few rows (of our 1000: German 1, Korean 6, Chinese 7; none elsewhere), and
MMMLU's Swahili and Yoruba files label some subjects differently; both counts are printed, and more than 2% of
rows disagreeing on the answer stops the script (that would mean the rows are not aligned).
Writes data/splits/mmmlu_<lang>_1000.json (same fields as mmlu_1000, plus mmlu_row and source).
Arabic GSM8K: none exists (MGSM has no Arabic).
Usage: python data/prepare_mmmlu.py            (every language; ar is rewritten identically)
       python data/prepare_mmmlu.py sw yo      (only these)
"""
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "splits"
HF = "https://huggingface.co/datasets"
MMMLU = {"ar": "AR-XY", "bn": "BN-BD", "de": "DE-DE", "es": "ES-LA", "fr": "FR-FR", "it": "IT-IT", "ja": "JA-JP",
         "ko": "KO-KR", "sw": "SW-KE", "yo": "YO-NG", "zh": "ZH-CN"}
GLOBAL_MMLU = ["nl", "pl", "ru", "vi"]


def get(path, csv=False):
    with urllib.request.urlopen(f"{HF}/{path}") as r:
        raw = io.BytesIO(r.read())
    return pd.read_csv(raw) if csv else pd.read_parquet(raw)


def main():
    langs = sys.argv[1:] or list(MMMLU) + GLOBAL_MMLU
    en = get("cais/mmlu/resolve/main/all/test-00000-of-00001.parquet")
    index = {}
    for i, (s, q, c) in enumerate(zip(en.subject, en.question, en.choices)):
        index.setdefault((s, q, tuple(c)), i)  # some questions repeat across subjects
    within = en.groupby("subject").cumcount()  # position within the subject, for Global-MMLU's sample_id
    questions = json.loads((OUT / "mmlu_1000.json").read_text())
    for lang in langs:
        if lang in MMMLU:
            tr = get(f"openai/MMMLU/resolve/main/test/mmlu_{MMMLU[lang]}.csv", csv=True)
            assert len(en) == len(tr), (lang, len(en), len(tr))
            source = "openai/MMMLU"
        elif lang in GLOBAL_MMLU:
            tr = get(f"CohereLabs/Global-MMLU/resolve/main/{lang}/test-00000-of-00001.parquet").set_index("sample_id")
            source = "CohereLabs/Global-MMLU"
        else:
            raise SystemExit(f"No MMLU translation for '{lang}' in MMMLU or Global-MMLU")
        rows, other_subject, other_answer = [], 0, 0
        for q in questions:
            i = index[(q["subject"], q["question"], tuple(q["choices"]))]
            if lang in MMMLU:
                r = tr.iloc[i]
                subject, answer, text = r.Subject, r.Answer, r.Question
                choices = [str(r[l]) for l in "ABCD"]
            else:
                r = tr.loc[f"{q['subject']}/test/{within[i]}"]
                subject, answer, text = r.subject, r.answer, r.question
                choices = [str(r[f"option_{l}"]) for l in "abcd"]
            assert q["subject"] == en.subject[i], (lang, i, q["subject"])
            other_subject += subject != q["subject"]
            other_answer += "ABCD".index(answer) != q["answer"]
            row = {"question": str(text), "choices": choices, "answer": "ABCD".index(answer), "subject": q["subject"],
                   "mmlu_row": i}
            rows.append(row if lang == "ar" else {**row, "source": source})  # ar: keep the file as first written
        assert other_answer <= 0.02 * len(rows), (lang, other_answer)
        (OUT / f"mmmlu_{lang}_1000.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(f"mmmlu_{lang}_1000: {len(rows)} ({source}); answer differs from the English key on {other_answer}, "
              f"subject label on {other_subject}")


if __name__ == "__main__":
    main()
