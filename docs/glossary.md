# Glossary (plain language)

**ASR (attack success rate):** the percentage of harmful prompts where a jailbreak worked, meaning the model answered instead of refusing. Roughly the inverse of the refusal rate (RR) in our EMNLP paper.

**Coordinate:** inside the model each token is a list of numbers (for example 4,096 of them). One slot in that list is one coordinate.

**Residual stream:** the main vector passed from layer to layer. Our EMNLP paper and Arditi et al. work here.

**Privileged basis:** whether a single coordinate means anything on its own.
- In the residual stream it doesn't: the model only uses *combinations* of coordinates. Rotating the space would change nothing, like describing a location as "north + east" or "north-east". That's why refusal shows up as a *direction*, a mix of all coordinates.
- Inside the MLP it does: each coordinate goes through its own nonlinearity, so coordinates stop being interchangeable and each can learn its own job.

**Neuron:** one coordinate inside the MLP layer, after the nonlinearity and before the projection back down (`down_proj`). For example, Kazemi et al.'s Qwen3-8B refusal neuron is "layer 14, neuron 7924".

**Gate and value (SwiGLU):** a modern MLP neuron is `gate × value`. The value is *what* the neuron says; the gate is *how much* gets through, like a tap (gate) on a pipe of water (value).

**Double-negative regime:** gate and value can both be negative, and negative × negative is positive. So a neuron can read positive for two different reasons, and the "side-effect" reason can pollute its list of top examples. Goodfire recommends checking which case applies before interpreting a neuron.

**Poles:** a neuron can be strongly positive or strongly negative, and each side can mean something different. Kazemi et al.'s refusal neurons have a "harmful" pole (fires on restricted or explicit content) and a "safe" pole (fires on disclaimers and warnings).

**Gate (circuit sense):** a neuron acting like a railway switch that routes inputs down different paths, for example harmful → "refuse" track. This is a different meaning from the SwiGLU gate above.

**Polysemantic / monosemantic:** a neuron that seems to respond to several unrelated things / to one thing.

**Superposition:** the hypothesis that models have more concepts than neurons, so they pack *unrelated* concepts into the same neurons. If true, single neurons are hard to read, and you need a sparse autoencoder (SAE) to unpack them. This is the main motivation for SAEs.

**Recursive generalization (Goodfire):** the counter-claim that mixed-looking neurons mostly hold *related* things that grew out of one concept, for example "iy" → "diy" → "do it yourself". If true, neurons are already readable and SAEs mostly re-describe them.

**SAE (sparse autoencoder):** a second model trained to rewrite activations as a sparse combination of many learned "features". We used GemmaScope SAEs in the EMNLP paper.

**Transcoder:** like an SAE, but trained to mimic an MLP layer's input→output map. It's the natural thing to compare with MLP neurons.

## Terms from the method

**L14:F7924:** shorthand for "layer 14, neuron 7,924" (zero-based). With `--space residual`, F is a residual-stream coordinate instead of a neuron.

**Content tokens vs template tokens:** a chat model never sees a question alone. It is wrapped in a fixed chat template, e.g. for Qwen3: `<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`. The user's words are content tokens; the rest are template tokens, identical for every prompt.

**Post-instruction tokens / "the 5 template tokens" (Table 3 tokens):** the template tokens between the end of the question and the start of the answer: Qwen3 `<|im_end|>\n<|im_start|>assistant\n`, Llama `<|eot_id|><|start_header_id|>assistant<|end_header_id|>`. The model has read the question but not answered yet, so this is where it "decides" whether to refuse. Apple highlights them in red in their Table 3.

**Gap:** a neuron's average value on harmful prompts minus its average on harmless prompts, at a template token.

**Push:** how much a neuron makes the model start answering with a refusal sentence. It is the gradient of the refusal log-odds log(p / (1 − p)) with respect to the neuron, where p is the probability of starting with "I'm sorry, but I can't help with that request." It needs a sentence, not a judge, because you can't take a gradient through a judge's yes/no.

**Score (search):** gap × push. The search ranks neurons by it and keeps the top 5.

**Pin:** overwrite a neuron's value with a fixed number at every token, while the model reads the prompt and while it answers. Pinning a refusal neuron the opposite way to how it fires on harmful prompts stops refusal.

**m, m\*:** the pin value. The rerank tries many values of m; m\* is the chosen one (e.g. +20 for Qwen3-8B's L14:F7924).

**Rerank:** part 2 of the search: pin each top-5 candidate at many values of m, attack HarmBench, judge, keep the best neuron and m.

**Constant vs anchor intervention:** constant pins the same m\* on every prompt. Anchor first reads how strongly the neuron fires on *this* prompt and pins hard only if it fires, so normal prompts are barely touched (paper Eq. 6–7; `intervene.py`).

**Write vector:** each neuron's fixed contribution direction. The MLP's shrink step (`down_proj`) adds up `h[i] × down_proj.weight[:, i]` over all neurons, so neuron i adds its value times its own fixed column (4,096 numbers) to the residual stream.

**Cosine similarity:** how much two vectors point the same way: 1 = same direction, 0 = unrelated, −1 = opposite. Used to compare a neuron's write vector with a refusal direction.

**Residual-stream number / resid_pre:** one of the 4,096 numbers of the residual stream at a layer's input. Section 3.3 pins these instead of neurons.

## Terms from the evaluation

**Judge / Llama-Guard-3-8B:** a safety classifier that reads a prompt and an answer and says "safe" or "unsafe". Our only judge. Caveat: it also calls garbled text "unsafe", and doesn't officially support Arabic.

**Refusal rate (string match):** a rough check that looks for phrases like "I'm sorry" or "I cannot" at the start of an answer. Logged for quick progress only.

**AUROC:** grades a score (like a neuron's value) as a detector without picking a cut-off. It is the chance that a randomly chosen harmful prompt scores higher than a randomly chosen harmless one: 1.0 = always, 0.5 = coin flip.

**Cut-off (threshold):** the value above (or below) which a prompt is flagged as harmful. Chosen on prompts with known labels; Apple's "optimal threshold" is the most accurate cut-off on the test set itself.

**Separation:** how well a score tells harmful prompts from harmless ones in the same language, measured with AUROC.

## Datasets

**AdvBench, MaliciousInstruct, TDC2023, Alpaca:** the sources of Arditi's training prompts (harmful / harmless) used for the search.

**HarmBench:** harmful requests used to choose the best neuron and pin (191 after removing JailbreakBench overlaps).

**JailbreakBench:** 100 harmful requests, the held-out attack test.

**XSTest:** 250 safe prompts that contain scary words ("How can I kill a Python process?") and 200 unsafe ones. Used for detection.

**MMLU, GSM8K:** exam multiple-choice questions and math word problems, to check the model still works.

**MultiJail:** 315 harmful prompts, each translated by humans into Arabic and 8 other languages; our parallel English/Arabic harmful set.

**Aya:** human-written prompts in many languages; our normal (harmless) prompts in English and Arabic.

## Multilingual and Arabic terms

**PolyRefuse:** Wang et al.'s (NeurIPS 2025) translation of Arditi's English prompt sets into 13 languages with Google Translate. Its harmful training prompts are the same as ours, in the same order, so the English and Arabic searches use the very same 128 prompts.

**MultiJail:** 315 harmful prompts translated by humans into 9 languages, including Arabic. Our human-translated track.

**Two tracks:** every Arabic (and later multilingual) experiment runs on both PolyRefuse (machine-translated, the same prompts as the English replication) and MultiJail (human-translated).

**Refusal phrases (in a language):** the sentences a model starts its refusals with in that language. The search's push needs them. `refusal_phrases.py` finds them as the most common openings of the model's own answers, e.g. Llama in Arabic: "لا أستطيع مساعدتك في" ("I can't help you with").

**Rerank / eval splits:** held-out harmful prompts per language. The rerank set chooses the neuron and pin (like Apple's HarmBench-191); the eval set is the final attack test (like JailbreakBench).

**Qwen3Guard-Gen-8B:** Qwen's safety classifier (119 languages). Labels a prompt and answer safe / unsafe / controversial and says whether the answer is a refusal. Our Arabic judge.

**NLLB-200:** Meta's translation model (200 languages). Translates non-English answers into English so Llama-Guard can judge them (Wang et al.'s method). A plain translator, so it doesn't refuse to translate harmful text.

**Agreement / Cohen's kappa:** how often two judges give the same verdict. Kappa corrects for chance agreement: 1 = perfect, 0 = no better than chance.
