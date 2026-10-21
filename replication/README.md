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
   - **Anchor:** scale the clamp by the prompt's own activation `v` (max or min over the post-instruction tokens) relative to the harmful–harmless gap `d`, capped at `m*`, with k ∈ {1, 2}. Re-derive the exact form from their Eq. 6–7 before coding it; the formula is garbled in text extraction.
5. **Evaluate:**
   - JailbreakBench (100 prompts), greedy decoding, 512 tokens.
   - Judges: Llama-Guard-3-8B plus an LLM judge (prompt in their Appendix K.1).
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
well below the MLP neurons. These are paper reference numbers, not our results;
the baseline is implemented but has not been run on the pretrained models here.

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
