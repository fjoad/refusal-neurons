"""Arabic MMLU: the same 1000 questions as mmlu_1000, from OpenAI's MMMLU (professional translations).

MMMLU's Arabic file (test/mmlu_AR-XY.csv) translates the MMLU test set row by row, in the same order as
cais/mmlu 'all'. Each mmlu_1000 question is found in the English set by its subject and text, then the same row
is taken from MMMLU; the subject and the answer letter must match on every row.
Writes data/splits/mmmlu_ar_1000.json (same fields as mmlu_1000). Arabic GSM8K: none exists (MGSM has no Arabic).
"""
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "splits"
HF = "https://huggingface.co/datasets"


def get(path, csv=False):
    with urllib.request.urlopen(f"{HF}/{path}") as r:
        raw = io.BytesIO(r.read())
    return pd.read_csv(raw) if csv else pd.read_parquet(raw)


def main():
    en = get("cais/mmlu/resolve/main/all/test-00000-of-00001.parquet")
    ar = get("openai/MMMLU/resolve/main/test/mmlu_AR-XY.csv", csv=True)
    assert len(en) == len(ar), (len(en), len(ar))
    index = {}
    for i, (s, q, c) in enumerate(zip(en.subject, en.question, en.choices)):
        index.setdefault((s, q, tuple(c)), i)  # some questions repeat across subjects
    rows = []
    for q in json.loads((OUT / "mmlu_1000.json").read_text()):
        i = index[(q["subject"], q["question"], tuple(q["choices"]))]
        r = ar.iloc[i]
        assert r.Subject == q["subject"] == en.subject[i], (i, r.Subject, q["subject"])
        assert "ABCD".index(r.Answer) == q["answer"], (i, r.Answer, q["answer"])
        rows.append({"question": str(r.Question), "choices": [str(r[l]) for l in "ABCD"],
                     "answer": q["answer"], "subject": q["subject"], "mmlu_row": i})
    (OUT / "mmmlu_ar_1000.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"mmmlu_ar_1000: {len(rows)}")


if __name__ == "__main__":
    main()
