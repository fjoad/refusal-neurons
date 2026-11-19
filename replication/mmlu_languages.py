"""MMLU accuracy per language, no intervention (17 November meeting, point 30).

The check behind dropping a language: if the model cannot answer MMLU in it, results in that language say
little about refusal. Every language gets the same 1000 questions (mmlu_1000 in English, mmmlu_<lang>_1000
translated; data/prepare_mmmlu.py). Scored as in capability.py: the next token among A/B/C/D after the chat
template. The answer instruction stays in English for every language, so only the question's language changes
(Arabic with an Arabic instruction is Arabic Experiment 3). Chance is 0.25.
Writes outputs/capability/<model>_mmlu_languages.json (predictions and accuracies) and logs accuracies, with the
share of each predicted letter (a model that always says one letter shows up there).
Usage: python replication/mmlu_languages.py --model Qwen/Qwen3-8B
       python replication/mmlu_languages.py --model Qwen/Qwen3-8B --langs sw yo
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from capability import LETTERS, mmlu_text
from neuron_check import ROOT, build_prompts, load

LANGS = ["en", "ar", "bn", "de", "es", "fr", "it", "ja", "ko", "nl", "pl", "ru", "sw", "vi", "yo", "zh"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left", local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto",
                                                 local_files_only=True)
    model.eval()
    letter_ids = [tok(l, add_special_tokens=False).input_ids[0] for l in LETTERS]
    out = ROOT / "outputs" / "capability" / f"{args.model.split('/')[-1]}_mmlu_languages.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"args": vars(args), "results": []}
    for lang in args.langs:
        data = load("mmlu_1000" if lang == "en" else f"mmmlu_{lang}_1000")
        prompts = build_prompts(tok, [mmlu_text(q) for q in data])[0]
        preds = []
        with torch.no_grad():
            for i in range(0, len(prompts), args.bs):
                batch = tok(prompts[i:i + args.bs], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
                logits = model(**batch, logits_to_keep=1).logits[:, -1, letter_ids]
                preds += logits.argmax(-1).tolist()
        acc = sum(p == q["answer"] for p, q in zip(preds, data)) / len(data)
        shares = [preds.count(k) / len(preds) for k in range(4)]
        result["results"].append({"lang": lang, "n": len(data), "accuracy": acc, "predicted_share": shares,
                                  "preds": preds})
        out.write_text(json.dumps(result, indent=1))
        print(f"{lang}: accuracy {acc:.3f} on {len(data)}; predicted A/B/C/D "
              + "/".join(f"{s:.2f}" for s in shares) + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"saved outputs/capability/{out.name}")


if __name__ == "__main__":
    main()
