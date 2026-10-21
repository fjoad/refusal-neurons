"""Llama-Guard-3 as a harmful-prompt detector, the baseline for the refusal neuron (paper Table 2).

Classifies each user prompt alone (no response). The score is P("unsafe") vs P("safe") at
the first non-whitespace token Llama-Guard generates; AUROC uses that score, and precision
and recall use Llama-Guard's own verdict. Prompts only, so nothing generated is printed.
Usage: python replication/guard_detect.py --harmful xstest_unsafe --harmless xstest_safe --tag xstest
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_check import ROOT, load

JUDGE = "meta-llama/Llama-Guard-3-8B"


@torch.no_grad()
def unsafe_scores(model, tok, prompts, bs):
    first = lambda w: tok(w, add_special_tokens=False).input_ids[0]
    id_unsafe, id_safe = first("unsafe"), first("safe")
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False) for p in prompts]
    scores, verdicts = [], []
    for i in range(0, len(chats), bs):
        batch = tok(chats[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**batch, max_new_tokens=5, do_sample=False, pad_token_id=tok.pad_token_id,
                             output_scores=True, return_dict_in_generate=True)
        gen = out.sequences[:, batch.input_ids.shape[1]:]
        for b in range(gen.shape[0]):
            step = next((t for t in range(gen.shape[1]) if tok.decode(gen[b, t]).strip()), 0)
            p = torch.softmax(out.scores[step][b].float(), -1)
            scores.append(float(p[id_unsafe] / (p[id_unsafe] + p[id_safe])))
            verdicts.append(tok.decode(gen[b], skip_special_tokens=True).strip().lower().startswith("unsafe"))
    return scores, verdicts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harmful", required=True, help="split name in data/splits/")
    ap.add_argument("--harmless", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(JUDGE, padding_side="left", local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(JUDGE, torch_dtype=torch.float16, device_map="auto",
                                                 local_files_only=True)
    model.eval()
    harmful, harmless = load(args.harmful), load(args.harmless)
    s_h, v_h = unsafe_scores(model, tok, harmful, args.bs)
    s_s, v_s = unsafe_scores(model, tok, harmless, args.bs)
    labels = np.r_[np.ones(len(s_h)), np.zeros(len(s_s))]
    auroc = roc_auc_score(labels, s_h + s_s)
    tp, fp = sum(v_h), sum(v_s)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / len(v_h)
    print(f"{JUDGE} on {args.harmful} vs {args.harmless}: AUROC {auroc:.3f}  precision {precision:.3f}  "
          f"recall {recall:.3f}  ({(time.time() - t0) / 60:.1f} min)")
    out = ROOT / "outputs" / "guard_detect"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps({
        "args": vars(args), "judge": JUDGE, "auroc": auroc, "precision": precision, "recall": recall,
        "scores_harmful": s_h, "scores_harmless": s_s, "verdicts_harmful": v_h, "verdicts_harmless": v_s,
    }, indent=1))


if __name__ == "__main__":
    main()
