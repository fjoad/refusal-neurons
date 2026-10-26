"""Machine translation into English with NLLB-200 (Meta), for judging non-English answers.

A plain translation model, not a chat model, so it translates harmful text instead of refusing
to (a chat model as translator could quietly turn unsafe answers into "safe" ones). Wang et al.
(NeurIPS 2025) judge non-English answers the same way, with Google Translate.
NLLB handles about 200 tokens per input, so each text is split at line and sentence breaks into
chunks, translated, and joined back with its line structure.
"""
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TRANSLATOR = "facebook/nllb-200-distilled-600M"
# Our language codes -> NLLB codes.
NLLB = {"en": "eng_Latn", "ar": "arb_Arab", "de": "deu_Latn", "es": "spa_Latn", "fr": "fra_Latn",
        "it": "ita_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang", "nl": "nld_Latn", "pl": "pol_Latn",
        "ru": "rus_Cyrl", "th": "tha_Thai", "yo": "yor_Latn", "zh": "zho_Hans", "vi": "vie_Latn",
        "bn": "ben_Beng", "sw": "swh_Latn", "jv": "jav_Latn"}


def chunks(text, max_chars=350):
    """Split at line breaks, then at sentence ends, packing pieces up to max_chars."""
    out = []
    for line in text.split("\n"):
        pieces, cur = re.split(r"(?<=[.!?؟。！？])\s*", line), ""
        for p in pieces:
            while len(p) > max_chars:  # very long run-on text: hard split
                out.append((cur + " " + p[:max_chars]).strip() if cur else p[:max_chars])
                cur, p = "", p[max_chars:]
            if len(cur) + len(p) + 1 > max_chars and cur:
                out.append(cur)
                cur = p
            else:
                cur = (cur + " " + p).strip()
        out.append(cur)
        out.append("\n")
    return out[:-1] if out else [""]


class Translator:
    def __init__(self, src_lang, device=None, bs=32):
        if src_lang not in NLLB:
            raise ValueError(f"No NLLB code for {src_lang}; add it to NLLB in translate.py")
        self.tok = AutoTokenizer.from_pretrained(TRANSLATOR, src_lang=NLLB[src_lang], local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR, local_files_only=True).half()
        self.model.to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
        self.bos = self.tok.convert_tokens_to_ids(NLLB["en"])
        self.bs = bs

    @torch.no_grad()
    def _translate(self, pieces):
        out = []
        for i in range(0, len(pieces), self.bs):
            batch = self.tok(pieces[i:i + self.bs], return_tensors="pt", padding=True, truncation=True,
                             max_length=256).to(self.model.device)
            ids = self.model.generate(**batch, forced_bos_token_id=self.bos, max_new_tokens=320,
                                      num_beams=1, do_sample=False)
            out += self.tok.batch_decode(ids, skip_special_tokens=True)
        return out

    def __call__(self, texts):
        """Translate a list of texts into English, keeping line breaks."""
        split = [chunks(t or "") for t in texts]
        todo = sorted({p for s in split for p in s if p.strip() and p != "\n"})
        done = dict(zip(todo, self._translate(todo))) if todo else {}
        result = []
        for s in split:
            text = ""
            for p in s:
                text += "\n" if p == "\n" else ((" " if text and not text.endswith("\n") else "") + done.get(p, p))
            result.append(text.strip())
        return result
