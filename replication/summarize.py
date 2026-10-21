"""Prints per-run metrics of a neuron_check.py or rerank.py output JSON (never the responses).

Shows string-match refusal, Llama-Guard ASR (if judge.py has run) and response lengths.
Usage: python replication/summarize.py outputs/rerank/<file>.json [more files...]
"""
import json
import statistics
import sys
from pathlib import Path

for path in sys.argv[1:]:
    d = json.loads(Path(path).read_text())
    runs = d["runs"] if isinstance(d["runs"], list) else [dict(r, pin=k) for k, r in d["runs"].items()]
    print(f"# {Path(path).name}: {len(d['prompts'])} prompts, judge={d.get('judge', 'none')}")
    for r in runs:
        where = f"L{r['layer']}:F{r['neuron']} " if r.get("layer") is not None else ""
        lens = [len(t) for t in r["responses"]]
        asr = f"  LG ASR {r['asr_llamaguard']:.2f}" if "asr_llamaguard" in r else ""
        print(f"  {where}pin={r['pin']}: string-match refusal {r['refusal_rate_stringmatch']:.2f}{asr}"
              f"  median length {statistics.median(lens):.0f} chars, empty {sum(n == 0 for n in lens)}")
