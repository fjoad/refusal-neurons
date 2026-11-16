"""CoCoNot splits (Brahman et al., 2024), the five categories as used in our EMNLP paper: for each category, the
prompts a model should not comply with ("refuse") and the matched contrast prompts it should answer ("comply").
Only "Requests with safety concerns" is about harm; the other four are refusals for other reasons (the request is
incomplete, unsupported, indeterminate, or treats the model as a person). Source: the EMNLP project's balanced files
(`exp1_v9/data/refusal_13splits_extended/CocoNot_cat_*`). Writes data/splits/coconot_<category>_{refuse,comply}.json.
Usage: /opt/homebrew/bin/python3.14 data/prepare_coconot.py "<path to>/llm safety project/exp1_v9/data/refusal_13splits_extended"
"""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "splits"
CATS = {"Requests_with_safety_concerns": "safety", "Humanizing_requests": "humanizing", "Incomplete_requests": "incomplete",
        "Indeterminate_requests": "indeterminate", "Unsupported_requests": "unsupported"}
src = Path(sys.argv[1])
for long, short in CATS.items():
    path = next(src.glob(f"CocoNot_cat_{long}_*.json"))
    rows = json.loads(path.read_text())
    refuse = [r["prompt"].strip() for r in rows if r["prompt_harm_label"] == "harmful"]
    comply = [r["prompt"].strip() for r in rows if r["prompt_harm_label"] != "harmful"]
    (OUT / f"coconot_{short}_refuse.json").write_text(json.dumps(refuse, indent=1, ensure_ascii=False))
    (OUT / f"coconot_{short}_comply.json").write_text(json.dumps(comply, indent=1, ensure_ascii=False))
    print(f"{short}: {len(refuse)} refuse, {len(comply)} comply")
