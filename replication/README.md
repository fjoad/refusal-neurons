# Step 0: replicate Kazemi et al. (2026)

Goal: recover their refusal neurons and approximate ASR with our own code. No official code is released, so we reimplement from Section 2 and Appendix A of [arXiv:2605.08513](https://arxiv.org/abs/2605.08513).

## Method to reimplement
1. **Hook:** read the pre-`down_proj` activation `h = SiLU(W_gate x) ⊙ W_up x` at every layer in the first 2/3 of the model. Each coordinate `h_i` is a neuron.
2. **Score:** compute the gradient of the refusal log-odds `L = −log(p_refusal / (1 − p_refusal))`, where `p_refusal` is the probability mass on the model-family refusal phrases (their Table 4). Take it at the post-instruction template tokens (their Table 3). Then:
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

## Data
- **Train:** 128 harmful (AdvBench, MaliciousInstruct, TDC2023) and 128 harmless (Alpaca) prompts, following Arditi et al.
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

Panther notes: run downloads as SLURM jobs (the login node kills them); install torch from PyPI; don't `module load` CUDA (it shadows torch's bundled libraries).

## Compute
Panther (QCRI SLURM). Only the 70B model needs multiple GPUs; start with the three models above.
