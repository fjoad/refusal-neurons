# Step 0: replicate Kazemi et al. (2026)

Goal: recover their refusal neurons and approximate ASR with our own code. No official code is released, so we reimplement from Section 2 and Appendix A of [arXiv:2605.08513](https://arxiv.org/abs/2605.08513).

## Method to reimplement
1. **Hook:** read the pre-`down_proj` activation `h = SiLU(W_gate x) ⊙ W_up x` at every layer in the first 2/3 of the model. Each coordinate `h_i` is a neuron.
2. **Score:** compute the gradient of the refusal log-odds `L = −log(p_refusal / (1 − p_refusal))`, where `p_refusal` is the probability mass on the model-family refusal phrases (their Table 4). Take it at the post-instruction template tokens highlighted red in their Table 3. For Qwen3 these are `<|im_end|>\n<|im_start|>assistant\n`, not the `<think>` block; for Llama they are `<|eot_id|><|start_header_id|>assistant<|end_header_id|>`. Then:
   - `G = mean gradient (harmful) + mean gradient (harmless)`
   - `score = G × (mean act harmful − mean act harmless)`
   - Take the best token per neuron.
   - Keep a neuron only if `|act harmful| > |act harmless|`.
3. **Rerank:** for the top 5 neurons, sweep the pinned value `m` (0–20 in steps of 2, then 20–200 in steps of 5) in the direction opposite to the neuron's harmful activation. Measure ASR on HarmBench (the 191 prompts after removing JailbreakBench overlap), and keep the best (layer, neuron, m*).
4. **Intervene:**
   - **Constant:** `h_i ← m*` at every token, during both prefill and generation.
   - **Anchor (Eq. 6–7, read from the rendered PDF):** `h_i ← clamp(k · m* · v/d − d, m*)`. Here `v` is the prompt's own max (if d > 0) or min (if d < 0) over the Table 3 tokens, from a hook-free pass, and `d = a_harmful − a_harmless` at the winning token. The clamp is min if m* > 0, max if m* < 0. `k ∈ {1, 2}`. Implemented in `intervene.py`.
5. **Evaluate:**
   - JailbreakBench (100 prompts), greedy decoding, 512 tokens.
   - Judges: Llama-Guard-3-8B plus an LLM judge (prompt in their Appendix K.1). **We use Llama-Guard-3 only** (see Deviations).
   - Capability cost: MMLU and GSM8K.
   - Detection: XSTest AUROC.

The Section 3.3 residual-stream baseline uses `neuron_search.py --space residual`:
the same gradient × activation score and filter, applied to individual dimensions at
each decoder layer's input (`resid_pre`) in the first two-thirds of layers. `L:F` and
the JSON's `neuron` field then mean layer:dimension. `rerank.py` infers the space from
the search JSON (or accepts a matching `--space residual`) and sweeps both +m and -m
for each top-5 candidate at every token. Search JSON, saved means and rerank output
names add `_residual`; `judge.py` works unchanged. Section 3.3 reports only about
45% ASR for the best Llama-3.1-8B residual feature (L12:F291) and 39% for Qwen3-8B,
well below the MLP neurons. Our results are in the log below ("Section 3.3"). Our hook
is each layer's *input*; the paper's L12:F291 matches our L13:F291, so the paper most
likely read each layer's *output*.

## Deviations from the paper

Everything else follows Section 2 and Appendices A, C and G as written.

| What | Paper | Us | Why / effect |
|---|---|---|---|
| Models | 7 (Qwen3 1.7B–32B, Llama-3.1 8B and 70B) | Qwen3-1.7B, Qwen3-8B, Llama-3.1-8B | GPU budget; covers both families |
| Precision, hardware | bf16 on one B200 | fp16 on V100/P100/T4, 8B models split over 2 cards | No fast bf16; fp16/bf16/fp32 give the same search ranking |
| Judge | Claude LLM judge (non-refusing *and* coherent) + Llama-Guard-3 | Llama-Guard-3 only | No API credits. Llama-Guard calls garbled text unsafe, so high-pin ASR is inflated and the Llama residual baseline can't be scored |
| Rerank grid | m = 0–20 step 2, 20–200 step 5 (47 values), 512 tokens | 9 values (2, 4, 6, 10, 15, 20, 30, 40, 60), 128 tokens | GPU time. Final attack (M3) uses 512 tokens like the paper |
| Rerank winner | highest HarmBench ASR | candidate reaching ASR ≥ 0.90 at the smallest pin | Because of the judge. Recovers Apple's neuron and m* on both 8B models |
| Qwen3-1.7B winner | L13:F3270 at +30 | L10:F244 at −15 (L13:F3270 at +40 also run in M3) | Near-tie among all 5 candidates, in the paper too |
| Capability sets | MMLU, GSM8K (size not stated) | 1,000 MMLU, 250 GSM8K, 256 new tokens for GSM8K | Time. Our GSM8K drops under the constant pin are larger than the paper's |
| Detection threshold | "optimal threshold" on XSTest | same (best accuracy on XSTest), plus a cut-off calibrated on other data | The optimal one is optimistic; calibrated numbers in the log |
| Residual baseline (3.3) | residual dims (hook point not stated) | each decoder layer's input | Paper's L12:F291 = our L13:F291, so they likely used layer outputs |
| Arditi refusal score | Arditi's pipeline | Arditi's rules re-implemented in `arditi.py` (refusal tokens "I", "As" for Qwen; "I" for Llama), Qwen3 template with thinking off | Arditi's code doesn't support Qwen3 |

## Data
- **Train:** 128 harmful (AdvBench, MaliciousInstruct, TDC2023) and 128 harmless (Alpaca) prompts: Arditi et al.'s exact sample (`arditi_*_train` splits). Our own seed-42 sample (`harmful_train`/`harmless_train`) gave different rankings.
- **Validation:** HarmBench minus the 9 prompts that overlap with JailbreakBench.
- **Test:** JailbreakBench.

## Targets to recover
| Model | Their neuron |
|---|---|
| Qwen3-1.7B | L13:F3270 |
| Qwen3-8B | L14:F7924 |
| Llama-3.1-8B-Instruct | L11:F4258 |

Qwen3 is run with thinking disabled.

## Results log

Newest work first in each topic; every entry has its date, setup, numbers and caveats. Index:
- **M1** (published neurons behave as described): 2026-10-19
- **Arabic teaser** (English neuron on parallel Arabic prompts): 2026-10-20
- **M2 search** (first run misses → three fixes → all three found): 2026-10-21
- **M2 rerank** (same neuron and pin as Apple on both 8B models): 2026-10-21
- **Judge sanity check** (Llama-Guard tracks the intervention, but also rewards garbled text): 2026-10-21
- **Detection** (XSTest matches Apple; neuron beats Llama-Guard in Arabic): 2026-10-21
- **Arabic search** (same top-5 neurons in English and Arabic): 2026-10-21, two entries
- **Per-token firing (3.1) and base models (3.2)**: 2026-10-21
- **Appendix G alignment and the Arditi baseline**: 2026-10-21
- **M3** (attack and capability cost): 2026-10-22
- **Section 3.3 residual baseline**: 2026-10-22
- **Refusal phrases in 24 language sets** (first pass): 2026-10-27
- **Arabic judges chosen and validated** (they agree; Llama-Guard on raw Arabic undercounts): 2026-10-27
- **Arabic Experiment 1** (search done; Qwen3-1.7B rerank a near-tie on PolyRefuse, under 0.90 on MultiJail; 8B reranks in progress): 2026-10-27

Job scripts that produced these are in `scripts/jobs/` (each says which entry it belongs to).

### 2026-10-27: Arabic Experiment 1, search results and Qwen3-1.7B rerank (8B reranks in progress) (`scripts/jobs/arabic_exp1.sh`, `scripts/jobs/judge_ar_rerank.sh`)
Apple's method run from scratch in Arabic, on two tracks:
- **PolyRefuse Arabic:** the same 128 harmful prompts as the English replication (machine-translated), plus PolyRefuse's 128 harmless.
- **MultiJail Arabic:** human-translated, plus Aya Arabic.

The push targets each model's own Arabic refusal sentences, found with `refusal_phrases.py` (whole-word openings):

| Model | Arabic refusal sentences (PolyRefuse / MultiJail) |
|---|---|
| Qwen3-1.7B | "أنا أرفض مساعدتكم في" ("I refuse to help you with") / "لا أستطيع أن أقدم" ("I can't provide") |
| Qwen3-8B | "أنا لا أستطيع مساعدتك", "أنا لا أستطيع تقديم" ("I can't help you", "I can't provide") / "أنا لا أستطيع مساعدتك" |
| Llama-3.1-8B | "لا أستطيع مساعدتك في", "لا أستطيع تقديم معلومات" ("I can't help you with", "I can't provide information"), both tracks |

**Search, PolyRefuse Arabic track** (top 5; Apple's neuron in **bold**):

| Model | Arabic top 5 | English neuron's rank: English search → Arabic search |
|---|---|---|
| Qwen3-1.7B | L10:F244, **L13:F3270**, L12:F2897, L12:F4580, L9:F3183 | 4 → **2** (its harmful/harmless activation in Arabic: −8.32 / −0.11). #1, L10:F244, won our English 1.7B rerank |
| Qwen3-8B | L17:F11229, L18:F7664, L16:F8474, **L14:F7924**, L15:F11168 | 5 → **4** (−3.42 / −0.17); 4 of 5 neurons shared with the English top 5 |
| Llama-3.1-8B | **L11:F4258**, L15:F9635, L10:F1084, L10:F9064, L10:F4789 | 3 → **1** (+0.72 / +0.01) |

**Search, MultiJail Arabic track** (top 5; Apple's neuron in **bold**):

| Model | Arabic top 5 | Apple's neuron's rank (harmful / harmless activation) |
|---|---|---|
| Qwen3-1.7B | **L13:F3270**, L15:F1109, L10:F244, L12:F4580, L12:F2897 | **1** |
| Qwen3-8B | L18:F7664, L16:F8474, **L14:F7924**, L17:F11229, L16:F5318 | **3** (−3.19 / −0.28) |
| Llama-3.1-8B | **L11:F4258**, L15:F9635, L10:F1084, L10:F9064, L13:F1358 | **1** (+0.56 / +0.01) |

On both tracks, Apple's neuron is in the Arabic top 5 for every model, and the 8B models' lists overlap heavily across tracks (Qwen3-8B: 4 of 5 shared; Llama: 4 of 5).

**Rerank:** answers for every candidate × 9 pin values on `poly_ar_harmful_rerank` (191) and `mj_ar_harmful_rerank` (87), into `outputs/rerank/<model>_float16_{poly,mj}_ar.json`, judged with both chosen Arabic judges (validated in the next entry) by `scripts/jobs/judge_ar_rerank.sh`. Logs: `arabic_exp1-436891/2/3`, `judge_ar-436940` (Qwen3-1.7B), `judge_ar-437035` (Qwen3-8B, running). Llama is still generating.

**Rerank, Qwen3-1.7B** (attack success rate; Q3G = Qwen3Guard on the Arabic, LG-tr = Llama-Guard on the NLLB translation; best pin per candidate = smallest pin within 0.02 of its max):

| Track | Unpinned (Q3G / LG-tr) | Best per candidate, Q3G | Best per candidate, LG-tr | Reaches 0.90? |
|---|---|---|---|---|
| PolyRefuse | 0.41 / 0.46 | L12:F4580 −30 0.92; **L13:F3270 +40 0.89**; L10:F244 −20 0.87; L9:F3183 +30 0.85; L12:F2897 −15 0.84 | L12:F2897 −40 0.89; **L13:F3270 +40 0.87**; L10:F244 −20 0.85; L12:F4580 −30 0.82; L9:F3183 +30 0.81 | Only L12:F4580 under Q3G |
| MultiJail | 0.16 / 0.13 | L15:F1109 +40 0.66; L12:F4580 −40 0.66; **L13:F3270 +60 0.59**; L10:F244 −15 0.53; L12:F2897 −30 0.53 | L15:F1109 +40 0.58; L12:F4580 −30 0.47; **L13:F3270 +40 0.43**; L10:F244 −15 0.41; L12:F2897 −60 0.40 | None |

- **PolyRefuse: a near-tie again, as in English.** All five candidates reach 0.81–0.92, and the two judges rank them differently. The 0.90 rule picks L12:F4580 under one judge and nothing under the other, so it doesn't settle a winner. Apple's neuron, L13:F3270, is 2nd under both judges at +40, the same pin where it peaked in English (0.93 there).
- **Qwen3-1.7B already complies on 41–46% of the PolyRefuse Arabic prompts unpinned** (38% on English HarmBench), which leaves little room between unpinned and pinned.
- **MultiJail: pinning helps but no candidate comes close to 0.90** (best 0.66 / 0.58, from 0.16 / 0.13 unpinned). Possible reasons, not yet checked: MultiJail's borderline prompts (see Open questions in the README), only 87 rerank prompts, and 128 tokens per answer.
- **At large negative pins Qwen3Guard's refusal label rises again** (e.g. L10:F244 at −60: refusal 0.81 on PolyRefuse, 0.93 on MultiJail), while the attack rate falls. These high pins likely break the model's output; not inspected.
- **The judges agree much less here than in the validation:** kappa 0.55 on PolyRefuse (82% agreement, 8,786 answers) and 0.51 on MultiJail (77%, 4,002), against 0.88 in the validation. The validation pooled clearly unpinned and clearly pinned 8B answers; the rerank is mostly in-between and partly garbled answers from a small model. Both judges are still reported side by side; a coherence check would help here too.
- Caveat: one small model, 128-token answers, 191 / 87 prompts.

### 2026-10-27: Arabic judges chosen and validated (`judge.py`, `translate.py`, `judge_agreement.py`)
- **Qwen3Guard-Gen-8B** judges the Arabic directly. It labels safe / unsafe / controversial and refusal yes / no.
- **Llama-Guard-3** judges an **NLLB-200** English translation (Wang et al.'s method). NLLB is a plain translation model, so it doesn't refuse to translate harmful text.
- Validation job `judge_validation-436889`, `scripts/jobs/judge_validation.sh`:
  - Qwen3-8B answers the 100 `poly_ar_harmful_eval` prompts, unpinned and with L14:F7924 at +20
  - the answers are judged four ways (Llama-Guard on Arabic, Llama-Guard on the translation, Qwen3Guard on Arabic, Qwen3Guard on the translation), with pairwise agreement and Cohen's kappa
  - English check: Qwen3Guard vs Llama-Guard on the English M3 outputs
- **Results** (attack success rate (ASR); 100 answers per row):

  | Judge | Unpinned | L14:F7924 at +20 |
  |---|---|---|
  | Llama-Guard on the Arabic | 0.07 | **0.84** |
  | Llama-Guard on the NLLB translation | 0.15 | 0.95 |
  | Qwen3Guard on the Arabic | 0.07 | 0.95 |
  | Qwen3Guard on the NLLB translation | 0.13 | 0.95 |

  - **The two chosen judges agree.** Qwen3Guard on the Arabic vs Llama-Guard on the translation: 94% agreement, Cohen's kappa 0.88, over all 200 answers. The other pairs that leave out raw-Arabic Llama-Guard: kappa 0.90 and 0.92.
  - **Llama-Guard on the raw Arabic misses harmful answers.** It finds 0.84 where the other three find 0.95, and has the lowest agreement with them (kappa 0.79–0.83). Arabic is not one of its supported languages, so we don't use it for Arabic.
  - **Translation adds a few unsafe verdicts to unpinned answers** (0.13–0.15 vs 0.07). The pinned numbers don't move. Qwen3Guard calls 79% of the unpinned Arabic answers refusals, and 4% of the pinned ones.
  - **The neuron jailbreaks in Arabic at the English pin.** Apple's L14:F7924 at +20, found on English, takes Qwen3-8B from 0.07 to 0.95 on Arabic prompts (both chosen judges).
  - **English check:** on the M3 JailbreakBench answers (400), Qwen3Guard and Llama-Guard agree 97%, kappa 0.93. Qwen3Guard gives 0.96 for the pin, 0.89 / 0.94 for the anchor at k = 1 / 2, and 0.03 unpinned. So Qwen3Guard tracks our English judge closely.
  - **Caveat:** one model, one neuron, 100 prompts. Agreement is pooled over unpinned and pinned answers, which are easy to tell apart.

### 2026-10-27: Refusal phrases in 24 language sets, first pass (`refusal_phrases.py`, `scripts/jobs/refusal_phrases.sh`)
Each model's two most common openings over 128 unpinned answers to every multilingual harmful train set: PolyRefuse in 14 languages, MultiJail in 10.

- **Llama-3.1-8B refuses with templated sentences in almost every language,** e.g.:
  - "Ich kann nicht dabei helfen" (German), "No puedo proporcionar…" (Spanish)
  - "Я не могу помочь вам" (Russian), "ขออภัย แต่" (Thai)
  - "Saya tidak bisa membantu" (Javanese), "Sina uwezo wa" (Swahili)
  - Exceptions: Yoruba (the openings echo the prompt), and no shared opening for Japanese or Korean.
- **Qwen3-8B refuses clearly in the high-resource languages** (German, Spanish, French, Italian, Polish, Russian, Vietnamese, Arabic, Chinese). For others it opens with hedges or explanations, e.g. Japanese "ご質問の内容について" ("regarding your question"), Thai "ขออธิบาย" ("let me explain"), Dutch "Het is belangrijk om te" ("it's important to"). Yoruba, Swahili, Bengali and Javanese show no clear refusal opening.
- **Qwen3-1.7B often doesn't refuse at all outside English and a few languages:**
  - Dutch opens with "Natuurlijk! Hier" ("Of course! Here")
  - Thai and Yoruba echo the prompt
  - Spanish, Korean and Russian have no shared opening

Takeaways:
- **The most common opening is not always a refusal,** so these phrases need a refusal check before they are used for search (planned: a local model labels each opening).
- **Low-resource languages get fewer templated refusals,** especially on the Qwen models. That's a first hint for direction 2 (why translation jailbreaks work).

Caveats:
- This first pass cut openings at 6 tokens, which can end mid-word (some Chinese and Bengali openings even end in broken characters). The script now cuts at whole words, so the 24-set run must be redone; only the Arabic sets have been, in the Arabic Experiment 1 entry above.
- Logs: `refusal_phrases-436876/7/8`; outputs `outputs/refusal_phrases/<model>_multilingual.json`.

### 2026-10-19: Milestone 1, published neurons behave as described (`neuron_check.py`)
Setup: JailbreakBench (100 prompts) vs held-out Alpaca (100) for separation. Bypass measured on the first 30 JailbreakBench prompts, greedy decoding, 96 new tokens, string-match refusal check (rough; not the paper's judge). V100-32GB.

| Model, neuron | AUROC | Refusal rate at pin = none / +10 / +20 / +40 / +80 | Notes |
|---|---|---|---|
| Qwen3-1.7B L13:F3270 (fp16) | 0.999 | 0.67 / 0.40 / 0.20 / 0.03 / 0.00 | Fires about −8 to −9 at the newline tokens after `<|im_end|>`, after `assistant`, and after `</think>`; harmless ≈ 0 |
| Qwen3-1.7B L13:F3270 (fp32) | 0.999 | identical | fp16 is fine on V100 (no bf16 needed) |
| Qwen3-8B L14:F7924 (fp16) | 0.997 | 0.97 / 0.53 / 0.03 / 0.00 / 0.00 | Peak at the newline after `<|im_end|>` (−4.0 vs −0.1). At +80 the output is gibberish, so we need a coherence-aware judge |
| Llama-3.1-8B-Instruct L11:F4258 (fp16) | 1.000 | 1.00 / 0.00 / 0.00 / 0.00 / 0.00 (pin is negative here) | Fires **positive** on harmful prompts (+1.0 vs 0.0 at the final `\n\n`), matching the paper's sign for Llama |

### 2026-10-20: Arabic teaser, same neuron on parallel English/Arabic prompts
Data (`data/prepare_arabic.py`): 100 MultiJail prompts, the same prompts in English and in human-translated Arabic, vs 100 human-written Aya prompts in the matching language. Same model and neuron (Qwen3-8B L14:F7924), with no search in Arabic yet. Refusal measured on the first 30 prompts with string-match; Arabic markers were added, but the check is rough.

| Language | AUROC | Peak token (newline after `<|im_end|>`): harmful vs harmless | Refusal rate at pin = none / +10 / +20 / +40 / +80 |
|---|---|---|---|
| English | 0.977 | −3.21 vs −0.24 | 0.47 / 0.23 / 0.17 / 0.00 / 0.00 |
| Arabic | 0.954 | −3.10 vs −0.27 | 0.43 / 0.17 / 0.10 / 0.07 / 0.00 |

Takeaways:
- **The English neuron transfers to Arabic.** It fires at the same token with nearly the same magnitude on the Arabic versions of the same prompts, and separation drops only slightly (0.977 → 0.954).
- **Intervening has the same effect in both languages** under this rough check.
- **Caveats:**
  - The baseline refusal rate is low (~0.45) in both languages. MultiJail comes from red-teaming data and includes many borderline prompts, and string-match misses some refusals.
  - We need a proper multilingual judge before these numbers mean anything.
  - The harmless sets are not parallel across languages.
  - n = 30 for refusal.

Panther notes: run downloads as SLURM jobs (the login node kills them); install torch from PyPI; don't `module load` CUDA (it shadows torch's bundled libraries).

### 2026-10-21: Milestone 2 passes, our search finds all three published neurons (`neuron_search.py`)
It took three fixes, each a place where our first version differed from the paper. Together they reproduce the paper's search:

| Run | Refusal phrases | Scored tokens | Training sample | Qwen3-1.7B L13:F3270 rank | Qwen3-8B L14:F7924 rank | Llama-3.1-8B L11:F4258 rank |
|---|---|---|---|---|---|---|
| 1 | 28 short markers (wrong) | whole suffix (wrong) | ours, seed 42 | not in top 5 | not in top 5 | — |
| 2 | Table 4 | whole suffix (wrong) | ours | 29 | 6 | — |
| 3 | Table 4 | Table 3 red tokens | ours | 11 (fp16), 10 (fp32), 11 (bf16) | 2 | 4 |
| **4** | **Table 4** | **Table 3 red tokens** | **Arditi's exact sample** | **4** | **5** | **3** |

Run 4 activations at the winning token match Table 5 closely, which suggests we now have the paper's training prompts:

| Neuron | Token | Ours, harmful / harmless | Table 5 |
|---|---|---|---|
| Qwen3-1.7B L13:F3270 | `\n` after `<|im_end|>` | −10.25 / +0.05 | −10.28 / −0.25 |
| Qwen3-8B L14:F7924 | `\n` after `<|im_end|>` | −4.41 / −0.16 | −4.41 / −0.23 |
| Llama-3.1-8B L11:F4258 | `<|eot_id|>` | +1.11 / −0.01 | +1.11 / −0.00 |

What each fix was:
- **Refusal phrases:** the paper's Table 4 uses two full phrases per family; for Qwen3 these are "I'm sorry, but I can't help with that request." and "I'm unable to assist". We had reused `neuron_check`'s 28 short markers, which include Arabic ones.
- **Scored tokens:** Table 3 highlights only some template tokens in red; we found this by rendering the PDF page, since text extraction drops the color. For Qwen3 they are `<|im_end|>\n<|im_start|>assistant\n`, not the empty `<think>` block. For Llama they are `<|eot_id|><|start_header_id|>assistant<|end_header_id|>`, without the final `\n\n`. Scoring the whole suffix let neurons at the last `\n\n` dominate, with huge activations such as −28.9 vs +0.4.
- **Training sample:** Arditi's pipeline uses `random.seed(42)`, then samples 128 from his `splits/harmful_train.json`, then 128 from `splits/harmless_train.json`. Our own sample shared only 25 of 128 harmful prompts with his. Precision (fp16/bf16/fp32) made no difference.

The published neuron is in the top 5 for every model, as the paper requires. It isn't always first there either: their Figure 3 rerank picks candidate 4 (1.7B), 3 (8B) and 4 (Llama). Our ranks among close scores (8B scores span −0.29 to −0.25) differ by one or two places. These are now the search defaults. Runs took 1.4–5.6 min on P100s, with logs `neuron_search-431393/4/5`.

**Next for M2:** the rerank. Sweep *m* for each top-5 candidate on HarmBench and choose the best attacker; this needs the judge.

### 2026-10-21: Milestone 2, first search run misses the published neurons (`neuron_search.py`)
Setup: 128 harmful / 128 harmless train prompts, first 2/3 of layers, fp16. V100-16GB (1 card for 1.7B, 2 for 8B); 30 and 49 min. **Refusal phrases were wrong:** the script defaulted to `neuron_check`'s 28 short refusal markers, which include 10 Arabic ones. The paper's Table 4 uses two full phrases per family; for Qwen3 these are "I'm sorry, but I can't help with that request." and "I'm unable to assist".

| Model | Top 5 (score; all at the final `\n\n` after `</think>` except where noted) | Published neuron |
|---|---|---|
| Qwen3-1.7B | L15:F1109 (−2.10), L17:F3328 (−1.38), L10:F244 (−0.93, first `\n`), L17:F419, L16:F2079 | L13:F3270 not in top 5 |
| Qwen3-8B | L23:F5088 (−2.27), L19:F8473 (−0.31), L22:F3309, L17:F11229 (first `\n`), L20:F5009 | L14:F7924 not in top 5 |

The top candidates have very large activation gaps; for example, L15:F1109 is −28.9 on harmful vs +0.4 on harmless. The published neurons are about −10 vs −0.25 (Table 5). Their full rank was not recorded because the script saved only the top 5.

Fixes in commit `347142b`:
- defaults to the Table 4 phrases
- `--watch L:F` reports a neuron's global rank
- saves the per-layer mean activations and gradients (`*_means.pt`), so other scorings can be checked without rerunning

The rerun and the later fixes are in the entry above.

### 2026-10-22: Milestone 3, attack and capability cost (`evaluate.py`, `judge.py`, `capability.py`)
Setup:
- **Attack:** JailbreakBench, 100 prompts, 512 new tokens, greedy, judged by Llama-Guard-3 only.
- **Capability:** fixed subsets of MMLU (1,000 questions; next-token A–D) and GSM8K (250 problems; 256 new tokens, answer after `####`), with the same chat template and intervention.
- **Interventions:**
  - constant pin `h ← m*` at every token
  - anchor Eq. 7, `h ← clamp(k·m*·v/d − d, m*)`, with v the prompt's own peak activation over Table 3's tokens
- **m\*:** from our rerank rule (first pin reaching ASR ≥ 0.90).

| Neuron, m* | JBB ASR: unpinned → constant / anchor k=1 / k=2 | MMLU: base → constant / anchor k=1 / k=2 | GSM8K: base → constant / anchor k=1 / k=2 |
|---|---|---|---|
| Qwen3-8B L14:F7924, +20 | 0.04 → **0.95** / 0.90 / 0.95 | 0.675 → 0.604 / **0.679** / 0.681 | 0.580 → 0.476 / **0.576** / 0.576 |
| Llama-3.1-8B L11:F4258, −4 | 0.04 → **0.93** / 0.94 / 0.93 | 0.638 → 0.607 / **0.638** / 0.638 | 0.728 → 0.540 / **0.756** / 0.748 |
| Qwen3-1.7B L13:F3270, +40 | 0.20 → 0.88 / 0.85 / 0.87 | 0.537 → 0.484 / 0.529 / 0.524 | 0.584 → 0.496 / 0.580 / 0.544 |
| Qwen3-1.7B L10:F244, −15 | 0.20 → 0.89 / 0.87 / 0.89 | 0.537 → 0.445 / 0.528 / 0.515 | 0.584 → 0.492 / 0.580 / 0.564 |
| Arditi ablation (entry below) | 0.04–0.18 → 0.91–0.95 | not run | not run |

Paper, under Llama-Guard: constant 91.9%, anchor 90.1%, Arditi 91.6% ASR on average. Capability cost (Table 1 averages): constant −8.8 MMLU / −1.2 GSM8K; anchor −0.6 / −0.1.

Takeaways:
- **ASR reproduces:** one pinned neuron jailbreaks both 8B models at 0.93–0.95, on par with Arditi's all-layer ablation.
- **The capability pattern reproduces:** the constant pin costs 3–9 MMLU points, and the anchor recovers essentially all of it (−0.9 to +0.6).
- **Our GSM8K drops under the constant pin are larger than the paper's** (−9 to −19). That may come from our 256-token limit plus pinning during the whole generation.

Caveats:
- Llama-Guard only. It doesn't check coherence, which matters for constant pins.
- Capability is measured on subsets, and Qwen3-8B's GSM8K base score (0.58) is low because thinking is disabled and generation is capped at 256 tokens.

Logs: `m3-431479..82`, `cap-431522/3`.

### 2026-10-22: Section 3.3 residual-stream baseline (`neuron_search.py --space residual`, `rerank.py`)
Setup: the same search on residual-stream coordinates at each decoder layer's *input*, first 2/3 of layers. The rerank pins one coordinate at every token, sweeping m ∈ {2, 4, 8, 15, 30, 60} in both directions, on 191 HarmBench prompts with Llama-Guard-3.

- **Qwen3-8B: the best residual dimension reaches only 0.26 ASR** (L19:F1214 at −60; unpinned 0.13). None reaches 0.90. Single residual dimensions are far weaker than the MLP neuron (0.97), which reproduces the paper's point (39%).
- **Llama-3.1-8B: Llama-Guard alone can't score this baseline.** Top dimensions (L13/L14/L15:F2742, L15:F325, L14:F2082) reach 0.92–0.98 at |pin| = 8, and L15:F325 and L14:F2082 do so in *both* directions. These dimensions only differ by about 0.3–0.5 between harmful and harmless prompts, so ±8 is about 16× their scale. The model is being scrambled, and Llama-Guard calls the garbled text unsafe. The paper's LLM judge requires coherence, which explains their 29–45%.
- **Hook point:** the paper's Llama residual top candidate L12:F291 is our **L13**:F291, the input of layer 13 = output of layer 12. It ranks 2nd there by score but fails our magnitude filter. So the paper most likely hooked each layer's *output*.

Next: a coherence-aware judge, i.e. a local LLM with the paper's K.1 prompt. Logs: `residual-431447/8`.

### 2026-10-21: Appendix G alignment and the Arditi ablation baseline (`arditi.py`)
Arditi's direction was selected with his pipeline rules on his exact 128/128 train and 32/32 val samples, filtered by refusal score:
- candidates: every layer × every post-instruction position
- discard the last 20% of layers, KL > 0.1, or induce < 0
- choose the lowest bypass score

The ablation projects the direction out at every layer.

| Model | Selected (pos, layer); paper's layer | Top \|cos\| neuron in that layer: ours / paper | E[max] | Published neuron's own-layer rank by cosine |
|---|---|---|---|---|
| Qwen3-8B | (−1, **20**); 20 ✅ | **F8719** 0.154 / F8719 0.200 ✅ | 0.070 | L14:F7924 **1st** (cos −0.149) |
| Llama-3.1-8B | (−1, **12**); 12 ✅ | **F5760** 0.198 / F5760 0.148 ✅ | 0.071 | L11:F4258 **1st** (cos +0.583) |
| Qwen3-1.7B | (−8, 15); 13 | F1109 0.174 / F3270 0.186 | 0.096 | L13:F3270 **1st** (cos −0.323) |

Union-bound p-values are ≤ 1e-11 for all three.

- **On both 8B models, the direction layer and its top neuron reproduce Table 7.**
- **Bonus check: in all three models, the published refusal neuron is the neuron in its own layer most aligned with Arditi's direction.** This convergence is stronger than the paper reports (2 of 7).
- **On 1.7B, our direction picks layer 15, and its top neuron is L15:F1109.** That's the same neuron that tops our 1.7B Arabic search and wins the 1.7B rerank on max ASR. 1.7B kept only 8 of 32 harmful val prompts after filtering, because it complies with many, so the selection is noisy.

Ablation on JailbreakBench (100 prompts, 512 tokens, Llama-Guard-3 ASR):

| Model | Unpinned | Arditi ablation |
|---|---|---|
| Qwen3-1.7B | 0.18 | 0.92 |
| Qwen3-8B | 0.04 | 0.91 |
| Llama-3.1-8B | 0.04 | 0.95 |

Paper's Llama-Guard average for Arditi: 91.6%. Logs: `arditi-431444/5/6`.

### 2026-10-21: Per-token firing (3.1) and base models (3.2) (`token_acts.py`)
Mean peak activation of each published neuron, on content tokens (the user's text) vs Table 3 tokens. Inputs: 100 JailbreakBench, 100 held-out Alpaca and 100 XSTest-safe prompts. XSTest-safe prompts contain harmful words with benign intent, like "kill a Python process".

| Model | Harmful: content / Table 3 | Harmless: content / Table 3 | XSTest-safe: content / Table 3 | AUROC harmful vs XSTest-safe (Table 3) |
|---|---|---|---|---|
| Qwen3-1.7B | −8.50 / **−11.22** | −0.07 / −0.40 | −3.54 / −4.65 | 0.927 |
| Qwen3-8B | −2.77 / **−4.11** | −0.09 / −0.09 | −1.46 / −1.14 | 0.919 |
| Llama-3.1-8B | +0.50 / **+1.31** | +0.02 / +0.01 | +0.29 / +0.28 | 0.958 |
| Qwen3-1.7B-**Base** | −6.00 / −5.60 | +0.30 / −0.40 | −1.11 / −0.65 | 0.990 |
| Qwen3-8B-**Base** | **−2.23** / −0.75 | +0.01 / +0.01 | −0.85 / −0.06 | 0.900 |

- **3.1 reproduced:** instruct models fire hardest at the turn boundary on harmful prompts and stay silent on harmless ones. Benign prompts with harmful words get a **partial** response, at roughly 30–40% of the harmful level, not silence. So "a murder of crows" is reduced, not zero, though separation stays high.
- **3.2 reproduced on Qwen3-8B:** the base checkpoint's neuron already separates harmful from harmless (AUROC 0.945 on content tokens), but it fires on the harmful *content*, not at the turn boundary. Alignment moves *when* it fires, as Apple says. Qwen3-1.7B-Base already fires at both.

Logs: `token_acts-431441/2`. Per-token arrays are in `outputs/token_acts/` for plotting.

### 2026-10-21: Milestone 2 rerank, the published neuron wins on both 8B models at the paper's m* (`rerank.py`, `judge.py`)
Setup:
- the top-5 search candidates (Arditi sample), each pinned at every token over m ∈ {2, 4, 6, 10, 15, 20, 30, 40, 60}, opposite to the neuron's harmful sign
- 191 HarmBench prompts, 128 new tokens, greedy, judged by Llama-Guard-3

Selection rule: **the candidate that reaches ASR ≥ 0.90 at the smallest |pin|.** Apple used max ASR under an LLM judge that also requires coherence. Llama-Guard alone rewards garbled high-pin output; for example, Llama runs at −20 to −40 give constant 255-character (repetitive) responses scored 0.95. So max ASR is misleading here.

| Model (unpinned ASR) | Candidates: first pin with ASR ≥ 0.90 | Winner | Paper |
|---|---|---|---|
| Qwen3-8B (0.13) | **L14:F7924 +20 (0.97)**; L16:F227 +30; L17:F11229 −30; L15:F11168 and L18:F7664 never | L14:F7924, +20 | L14:F7924, m* = +20 ✅ |
| Llama-3.1-8B (0.15) | **L11:F4258 −4 (0.94)**; L15:F9635 +15; L20:F9928 −20; L17:F13532 +30; L20:F9424 −60 | L11:F4258, −4 | L11:F4258, m* = −4 ✅ |
| Qwen3-1.7B (0.38) | L10:F244 −15 (0.93); L15:F1109 +30; L12:F4580 −30; L12:F3582 +30; **L13:F3270 +40 (0.93)** | L10:F244, −15 | L13:F3270, m* = +30 |

- **Both 8B models:** the published neuron and its m* are recovered exactly.
- **Qwen3-1.7B is a near-tie** among all five candidates: 0.93–0.97 at the top, with similar curves. The paper's Figure 3 also shows all five 1.7B candidates within 74–77%. 1.7B already complies on 38% of HarmBench unpinned.
- **The other candidates' curves are erratic and non-monotonic,** with response lengths jumping between ~100 and ~640 characters. That's a sign they break the model rather than switch off refusal cleanly.

M3 runs on all four selected neurons, including both 1.7B candidates (jobs `m3-431479..82`). Full curves: `sbatch scripts/cpu.sh replication/summarize.py outputs/rerank/<model>_float16_arditi.json`. `judge.py` now prints this rule's ranking.

### 2026-10-21: Detection, the neuron vs Llama-Guard-3 (paper Section 3.4) (`neuron_check.py`, `guard_detect.py`)
Neuron score: min (or max) activation over Table 3's tokens, as in the paper. Llama-Guard-3-8B classifies the prompt alone, scored by P(unsafe).

| Data | Qwen3-1.7B L13:F3270 | Qwen3-8B L14:F7924 | Llama-3.1-8B L11:F4258 | Llama-Guard-3 (AUROC; precision / recall) |
|---|---|---|---|---|
| XSTest (200 unsafe / 250 safe) | 0.852 | 0.901 | 0.963 (paper 0.97) | 0.975; 0.96 / 0.82 (paper recall 0.83) |
| MultiJail-en vs Aya-en (100 / 100) | 0.962* | 0.972 | 0.991 | 0.967; 1.00 / 0.58 |
| MultiJail-**ar** vs Aya-ar (100 / 100) | 0.898 (0.908 whole suffix) | **0.956** | 0.927 | **0.910**; 0.97 / 0.57 |

\* This is the whole-suffix AUROC; the job ran before the Table 3 AUROC was added.

**Turning the score into a filter** (`detect_threshold.py`, added 2026-10-22). A prompt is flagged when the neuron's strongest value over the Table 3 tokens passes a cut-off.

| | Optimal cut-off on XSTest (as the paper) | XSTest accuracy, ours (paper) | Cut-off calibrated on MultiJail-en | its accuracy: MultiJail-en / MultiJail-ar / XSTest |
|---|---|---|---|---|
| Qwen3-1.7B | ≤ −7.7 | 79.3% (79.1%) | — | — |
| Qwen3-8B | ≤ −2.80 | 82.7% (82.7%) | ≤ −1.31 | 95.5% / 88.5% / 68.0% |
| Llama-3.1-8B | ≥ +0.46 | 89.1% (90.2%) | ≥ +0.106 | 97.5% / 81.0% / 64.7% |

- **The paper's per-model XSTest numbers reproduce within about a point:** AUROC 0.853 / 0.902 / 0.969 in the paper vs ours 0.852 / 0.901 / 0.963.
- **A cut-off transfers from English to Arabic** with some loss.
- **It does *not* transfer to XSTest's tricky prompts:** recall goes to about 1, precision collapses. The cut-off must be calibrated on prompts like the ones being filtered.

Takeaways:
- **In Arabic, one neuron of Qwen3-8B (0.956) beats Llama-Guard-3 (0.910) as a harmful-prompt detector.** Llama-Guard-3 doesn't officially support Arabic.
- Its recall at its own threshold is low in both languages (0.57–0.58): MultiJail has many borderline prompts.
- Llama's XSTest AUROC and Llama-Guard's recall match the paper.

Caveats: n = 100 per side for MultiJail, and Aya harmless prompts are not parallel across languages. Logs: `detect-431437`, `arabic-431433/4`.

### 2026-10-21: Arabic search on all three models: largely the same top-5 neurons in English and Arabic (`neuron_search.py`)
Same splits as the Qwen3-8B entry below: 128 parallel MultiJail en/ar prompts vs Aya train. Each Arabic search targets that model's own Arabic refusal phrases, taken from 100 unpinned Arabic responses with `refusal_openings.py`:
- Qwen3-1.7B: "لا أستطيع أن أقدم", "أنا لا أستطيع"
- Llama: "لا أستطيع تقديم معلومات" (31), "لا أستطيع مساعدتك في" (10)
- Qwen3-8B: "أنا لا أستطيع مساعدتك"

Top 5 of each search (Apple's neuron in **bold**). The English search uses Apple's phrases on MultiJail-en vs Aya-en; the Arabic search uses the model's own Arabic phrases on the same prompts in Arabic.

| Rank | Qwen3-8B, en | Qwen3-8B, ar | Llama, en | Llama, ar | Qwen3-1.7B, en | Qwen3-1.7B, ar |
|---|---|---|---|---|---|---|
| 1 | **L14:F7924** | L18:F7664 | L20:F9928 | **L11:F4258** | L12:F4580 | **L13:F3270** |
| 2 | L16:F5318 | L16:F8474 | L20:F9424 | L15:F9635 | L15:F1109 | L10:F244 |
| 3 | L16:F227 | **L14:F7924** | L15:F9635 | L10:F1084 | L10:F244 | L15:F1109 |
| 4 | L16:F8474 | L17:F11229 | **L11:F4258** | L10:F9064 | **L13:F3270** | L12:F4580 |
| 5 | L18:F7664 | L16:F5318 | L17:F13532 | L13:F1358 | L9:F3183 | L12:F5880 |
| Shared | 4 of 5 | | 2 of 5 | | 4 of 5 | |

Apple's neuron's rank and value per search:

| Model, published neuron | English search rank | Arabic search rank, own Arabic phrases | Arabic search rank, English phrases | Harmful / harmless activation at the `\n` after `<|im_end|>` (Qwen) or `<|eot_id|>` (Llama), en → ar |
|---|---|---|---|---|
| Qwen3-1.7B L13:F3270 | 4 | **1** | 568 | −6.44 / −0.25 → **−5.59 / −0.30** |
| Qwen3-8B L14:F7924 | 1 | 3 | **1** | −3.24 / −0.27 → −3.19 / −0.28 |
| Llama-3.1-8B L11:F4258 | 4 | **1** | **1** | +0.50 / +0.01 → +0.56 / +0.01 |

- **Largely the same refusal neurons come out in both languages:** 4 of 5 shared on both Qwen models, 2 of 5 on Llama. Apple's neuron is in both top-5 lists for all three models: #1 in Arabic for Llama and Qwen3-1.7B, #3 for Qwen3-8B (#1 with the English phrases). It fires at the same token with nearly the same activation as in English.
- **Only the search (part 1) is done in Arabic.** The rerank needs a judge that reads Arabic.
- **The refusal phrases matter.** Qwen3-1.7B's neuron drops to rank 568 when the search targets English refusal phrases, which the model doesn't produce in Arabic. Under that scoring its best token moves to `<|im_end|>`, where it fires only −1.07.
- **Correction:** an earlier version of this entry read the rank-568 run as "the neuron barely fires in Arabic". It compared activations at different tokens and was wrong.
- **Unpinned refusal is lower in Arabic** (string match, rough): 1.7B 0.40 en / 0.20 ar; Llama 0.77 en / 0.58 ar.

Logs: `arabic-431433/4` (English-phrase runs), `search_ar2-431457` (own-phrase runs).

### 2026-10-21: Arabic search on Qwen3-8B finds the same neuron (`neuron_search.py`)
Data: 128 MultiJail prompts, parallel English and human-translated Arabic, kept separate from the 100 teaser prompts. Harmless: 128 Aya *train* prompts per language. Paper settings otherwise (Table 3 tokens). The Arabic refusal phrase "أنا لا أستطيع مساعدتك" comes from `refusal_openings.py`: it opens 10 of 30 unpinned Arabic responses.

| Search | Refusal phrases | L14:F7924 rank | Its activation, harmful / harmless | Other top-5 |
|---|---|---|---|---|
| English | Table 4 (English) | **1** | −3.24 / −0.27 | L16:F5318, L16:F227, L16:F8474, L18:F7664 |
| Arabic | "أنا لا أستطيع مساعدتك" | **3** | −3.19 / −0.28 | L18:F7664, L16:F8474, L17:F11229, L16:F5318 |
| Arabic | Table 4 (English) | **1** | −3.19 / −0.28 | L16:F8474, L17:F11229, L16:F5318, L18:F7664 |

All winners peak at the `\n` after `<|im_end|>`.

Takeaways:
- **A search run entirely in Arabic recovers the English refusal neuron**, at the same token and with nearly the same activation.
- **The candidate pools overlap:** L16:F8474, L16:F5318 and L18:F7664 appear in both the English and Arabic top 5.

Caveats:
- One model and n = 128.
- The harmless sets are not parallel across languages.
- The rerank (ASR) for the Arabic candidates is not done; it needs an Arabic-capable judging step.

Logs: `search_ar-431406`.

### 2026-10-21: Llama-Guard-3 judge sanity check (`judge.py`)
These are M1's Qwen3-8B JailbreakBench outputs: 30 prompts, 96 new tokens, pinning L14:F7924.

| Pin | none | +10 | +20 | +40 | +80 |
|---|---|---|---|---|---|
| String-match refusal | 0.97 | 0.53 | 0.03 | 0.00 | 0.00 |
| Llama-Guard ASR | 0.03 | 0.50 | 0.97 | 0.93 | 1.00 |

The judge tracks the intervention cleanly. The caveat is pin +80, where outputs shrink to a median of 100 characters and are likely garbled, yet Llama-Guard still calls them unsafe; see the judge decision in the README. A smoke test of `rerank.py` on 8 HarmBench prompts ran end to end. Qwen3-1.7B unpinned refused none of those 8, and none of them are copyright items; the full run gives the real baseline.

## Compute
Panther (QCRI SLURM). Only the 70B model needs multiple GPUs; start with the three models above.
