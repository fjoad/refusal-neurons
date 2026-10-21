"""Builds fixed capability subsets for measuring what a neuron intervention costs (paper Table 1).

mmlu_1000:  1000 random MMLU test questions (Hendrycks et al. 2021, MIT), all subjects.
gsm8k_250:  250 random GSM8K test problems (Cobbe et al. 2021, MIT); answer = final number.
The paper does not say whether it used the full sets; subsets keep runs short on our GPUs.
"""
import io
import json
import random
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "splits"
HF = "https://huggingface.co/datasets"


def parquet(path):
    with urllib.request.urlopen(f"{HF}/{path}") as r:
        return pd.read_parquet(io.BytesIO(r.read()))


def main():
    rng = random.Random(42)
    mmlu = parquet("cais/mmlu/resolve/main/all/test-00000-of-00001.parquet")
    rows = [{"question": q, "choices": list(c), "answer": int(a), "subject": s}
            for q, c, a, s in zip(mmlu.question, mmlu.choices, mmlu.answer, mmlu.subject)]
    gsm = parquet("openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet")
    probs = [{"question": q, "answer": a.split("####")[-1].strip().replace(",", "")}
             for q, a in zip(gsm.question, gsm.answer)]
    for name, data in [("mmlu_1000", rng.sample(rows, 1000)), ("gsm8k_250", rng.sample(probs, 250))]:
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
        print(f"{name}: {len(data)}")


if __name__ == "__main__":
    main()
