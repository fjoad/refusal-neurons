"""Per-search counts used by the chance baseline: neurons that pass the magnitude filter (eligible for a top-k list)."""
import json
import numpy as np
from compare_search import all_scores, ranks
from core import OUT

out = {}
for name in ("text", "speech_v1"):
    s, _ = all_scores(name)
    out[name] = int(np.isfinite(s).sum())
sa, _ = all_scores("text"); sb, _ = all_scores("speech_v1")
both = np.isfinite(sa) & np.isfinite(sb)
out["both"] = int(both.sum()); out["total"] = int(len(sa))
for k in (1000, 10000):
    top = set(np.argsort(sa)[:k]) | set(np.argsort(sb)[:k])
    idx = np.array(sorted(i for i in top if both[i]))
    out[f"spearman_top{k}"] = float(np.corrcoef(ranks(sa[idx]), ranks(sb[idx]))[0, 1])
print(json.dumps(out))
(OUT / "search/filter_counts.json").write_text(json.dumps(out, indent=1))
