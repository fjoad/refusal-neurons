# What the Apple paper did, section by section

A careful read of Kazemi et al. (2026, arXiv 2605.08513), main text and every appendix (A–L). It answers Nadir's question at the 22 October meeting: before we build on the paper, did they already do the "direction inside the MLP" idea, or anything else we were about to do? Done 2026-10-27.

**Short answer: no.** The paper never builds a mean-difference direction over the MLP's neurons. It never decomposes such a direction, and never tests whether Arditi's direction works *through* the refusal neuron. The closest thing is Appendix G. It takes Arditi's *residual-stream* direction at one layer and ranks that layer's neurons by how well each neuron's write vector lines up with it. That is our Experiment 7, which we reproduced.

## Main text

| Section | What they did | Our status |
|---|---|---|
| 2.1–2.3 Finding the neuron | Gradient × activation score at the Table 3 template tokens, refusal phrases from Table 4, first 2/3 of layers, top 5, then a rerank by HarmBench ASR over a sweep of pin values | Reproduced (M2) |
| 2.3 Anchor intervention | Eq. 6–7: pin scaled by the prompt's own activation, clamped at m\*, k ∈ {1, 2} | Reproduced (M3) |
| 2.4 Attack and capability | JailbreakBench ASR under constant, anchor and Arditi; MMLU and GSM8K | Reproduced with Llama-Guard only |
| 3.1 Token-level readability | Per-token activations; fires at the template tokens; "a murder of crows" stays silent | Reproduced. Our XSTest control shows a *partial* response to scary words |
| 3.2 Base models | Same neuron in four Qwen3 base checkpoints; fires on harmful content tokens instead of the template | Reproduced on Qwen3-1.7B-Base and 8B-Base |
| 3.3 Residual stream vs MLP | Same search on residual dimensions: best 45% (Llama), 39% (Qwen3-8B) | Qwen3-8B reproduced (26%); Llama needs a coherence judge |
| 3.4 Detection | Neuron as a detector on XSTest at the "optimal threshold"; vs Llama-Guard-3 | Reproduced within about 1 point |
| 3.5 What neurons respond to | Max-activating text from 20K examples of The Pile (uncopyrighted subset), both poles | Not done (explicit content; a human would have to read it) |
| 4 Suicide neurons | Neurons found by inspecting max-activating text; *additive* hook h ← h + m on 20 benign prompts; Claude judges mention, coherence, relevance | Not done; not our question |

## Appendices

| Appendix | Content | Notes for us |
|---|---|---|
| **A** Setup | bf16, one B200; Arditi's 128/128 train, HarmBench-191, JailbreakBench; Table 3 template tokens (red); Table 4 refusal phrases; pin sweep 0–20 step 2, 20–200 step 5; anchor k ∈ {1, 2}; MMLU via A–D log-probs, GSM8K by greedy generation; Table 5 neuron details | All followed (deviations listed in replication/README.md) |
| **B** Related work | Refusal directions (incl. our EMNLP paper "questioning whether a single direction is truly sufficient"); neuron attacks (NeuroStrike, pruning, safety knowledge neurons); Chen et al. 2024 safety neurons in base models | Their framing: sets of neurons → "the lower bound is one" |
| **C** Detection | Table 6: all 7 models, AUROC / accuracy / F1 / precision / recall at the optimal threshold; single-token vs multi-token (min/max) aggregation; Llama-Guard scored from its safe/unsafe token probabilities | Reproduced. The "optimal threshold" is chosen on XSTest itself |
| **D** Activation distributions | Harmful vs harmless histograms of each refusal neuron (z-scored), instruct and base | Our AUROCs cover the same ground |
| **E** Residual vs MLP features | Top-5 residual candidates for Llama and Qwen3-8B; best 45% / 39%; poor separation in the residual stream | Done for Qwen3-8B |
| **F** Per-token activations | Per-token plots across Qwen3 sizes and examples | Done (Experiment 4) |
| **G** Geometric alignment | Arditi direction at its layer ℓ; cosine with each down_proj column; top neuron, E[max] under a random null, union-bound p-value (Eq. 9); same neuron as the search in 2 of 7 models | Reproduced; we also rank the neuron in its own layer (#1 in all 3) |
| **H** Attack and capability tables | Table 8: HarmBench and JBB ASR under both judges, constant / anchor / Arditi, per model. Table 9: MMLU / GSM8K for constant / anchor / **Arditi** | Our M3 matches. We haven't measured Arditi's capability cost (their −0.3 MMLU, −0.5 GSM8K) |
| **I** What neurons respond to | Harmful pole = explicit or age-restricted content; safe pole = meta-language about restrictions (disclaimers, HIPAA, prosecutions). Qwen3-8B's neuron has no coherent safe pole. Qwen3-14B has 3 independently sufficient refusal neurons | Multiple gates matter for the per-language story |
| **J** Intervention examples | Qualitative: bypass on a harmful prompt; **inducing refusal on a benign prompt** by pinning a neuron the other way (Qwen3-14B, α = −60); suicide-neuron poems | Inducing refusal is shown only qualitatively. Measuring it is a cheap extension (Arditi's "induce" test, but with one neuron) |
| **K** Prompts | K.1 the LLM judge prompt: returns is_refusal, is_coherent, provides_helpful_info with confidences; a success = non-refusing **and** coherent. K.2 the concept-steering judge | Use K.1 for our free local coherence judge (next step 5) |
| **L** Feature profiles | Top-15 max-activating Pile snippets per neuron, both poles, ~3,400 lines; largely explicit content | Not read in detail |

## What this means for our next steps

1. **The MLP-space direction is new.** Building it, decomposing it and steering with it is not in the paper. Appendix G is the nearest piece, and it works from the residual direction.
2. **"Does Arditi's direction work through this neuron?" is new.** The paper shows the two methods converge geometrically (G), never causally.
3. **Measuring "induce refusal" with the neuron is a cheap add-on.** Pinning toward the harmful pole on harmless prompts is only shown qualitatively in Appendix J.
4. **Multiple refusal neurons per model are expected** (Qwen3-14B has 3). For the multilingual story, per-language or per-family neurons are plausible and not a failure.
5. **Their LLM judge requires coherence** (K.1). That is the piece missing from our Llama-Guard-only numbers.
6. **Not in the paper:** any language other than English, any modality other than text, and released code.
