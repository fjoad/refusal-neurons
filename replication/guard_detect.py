"""A guard model as a harmful-prompt detector, the baseline for the refusal neuron (paper Table 2).

Classifies each user prompt alone (no response). Prompts only, so nothing generated is printed.
--guard llamaguard (default): Llama-Guard-3. The score is P("unsafe") vs P("safe") at the first
  non-whitespace token it generates; AUROC uses that score, precision and recall use its own verdict.
--guard qwen3guard: Qwen3Guard-Gen-8B (119 languages). It labels a prompt Safe / Controversial / Unsafe;
  the score is 0 / 0.5 / 1 (so its AUROC has only three levels), the verdict is Unsafe, and the strict
  verdict also counts Controversial. Output file: <tag>_qwen3guard.json.
Several sets can be given at once (equal-length lists), so the guard is loaded once.
Usage: python replication/guard_detect.py --harmful xstest_unsafe --harmless xstest_safe --tag xstest
       python replication/guard_detect.py --guard qwen3guard --harmful poly_de_harmful_test poly_fr_harmful_test \
           --harmless poly_de_harmless_test poly_fr_harmless_test --tag poly_de poly_fr
"""
import argparse
import json
import re
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_check import ROOT, load

GUARDS = {"llamaguard": "meta-llama/Llama-Guard-3-8B", "qwen3guard": "Qwen/Qwen3Guard-Gen-8B"}


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


@torch.no_grad()
def label_scores(model, tok, prompts, bs):
    """Qwen3Guard-Gen on the prompt alone: its Safety label as a score (Safe 0, Controversial 0.5, Unsafe 1)."""
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False) for p in prompts]
    labels = []
    for i in range(0, len(chats), bs):
        batch = tok(chats[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**batch, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        for text in tok.batch_decode(gen[:, batch.input_ids.shape[1]:], skip_special_tokens=True):
            m = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", text)
            labels.append(m.group(1) if m else "Unparsed")
    return labels


def pr(v_h, v_s):
    tp, fp = sum(v_h), sum(v_s)
    return (tp / (tp + fp) if tp + fp else float("nan")), tp / len(v_h)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harmful", required=True, nargs="+", help="split name(s) in data/splits/")
    ap.add_argument("--harmless", required=True, nargs="+")
    ap.add_argument("--tag", required=True, nargs="+", help="one output name per pair of splits")
    ap.add_argument("--guard", default="llamaguard", choices=sorted(GUARDS))
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    if not len(args.harmful) == len(args.harmless) == len(args.tag):
        ap.error("--harmful, --harmless and --tag need the same number of values")

    t0 = time.time()
    judge = GUARDS[args.guard]
    tok = AutoTokenizer.from_pretrained(judge, padding_side="left", local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(judge, torch_dtype=torch.float16, device_map="auto",
                                                 local_files_only=True)
    model.eval()
    out = ROOT / "outputs" / "guard_detect"
    out.mkdir(parents=True, exist_ok=True)
    for h_split, s_split, tag in zip(args.harmful, args.harmless, args.tag):
        harmful, harmless = load(h_split), load(s_split)
        res = {"args": {"harmful": h_split, "harmless": s_split, "tag": tag, "guard": args.guard, "bs": args.bs},
               "judge": judge}
        if args.guard == "llamaguard":
            s_h, v_h = unsafe_scores(model, tok, harmful, args.bs)
            s_s, v_s = unsafe_scores(model, tok, harmless, args.bs)
            extra, name = "", f"{tag}.json"
        else:
            l_h, l_s = label_scores(model, tok, harmful, args.bs), label_scores(model, tok, harmless, args.bs)
            level = {"Unsafe": 1.0, "Controversial": 0.5}
            s_h, s_s = [level.get(x, 0.0) for x in l_h], [level.get(x, 0.0) for x in l_s]
            v_h, v_s = [x == "Unsafe" for x in l_h], [x == "Unsafe" for x in l_s]
            p2, r2 = pr([x > 0 for x in s_h], [x > 0 for x in s_s])
            unparsed = sum(x == "Unparsed" for x in l_h + l_s)
            extra = f"  strict precision {p2:.3f} recall {r2:.3f}" + (f"  unparsed {unparsed}" if unparsed else "")
            res.update(labels_harmful=l_h, labels_harmless=l_s, precision_strict=p2, recall_strict=r2)
            name = f"{tag}_qwen3guard.json"
        labels = np.r_[np.ones(len(s_h)), np.zeros(len(s_s))]
        auroc = roc_auc_score(labels, s_h + s_s)
        precision, recall = pr(v_h, v_s)
        acc = (sum(v_h) + len(v_s) - sum(v_s)) / (len(v_h) + len(v_s))
        print(f"{judge} on {h_split} vs {s_split}: AUROC {auroc:.3f}  acc {acc:.3f}  precision {precision:.3f}  "
              f"recall {recall:.3f}{extra}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        res.update(auroc=auroc, accuracy=acc, precision=precision, recall=recall, scores_harmful=s_h,
                   scores_harmless=s_s, verdicts_harmful=v_h, verdicts_harmless=v_s)
        (out / name).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
