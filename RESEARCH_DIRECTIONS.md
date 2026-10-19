# Research directions

Written down on 2026-10-19 after a meeting with Nadir Durrani. Nadir's framing: build on Kazemi et al. (2026) for **multilingual and multimodal safety refusal**.

The terms used here (neuron, pole, gate, ASR) are explained in [docs/glossary.md](docs/glossary.md).

## Step 0: replicate Kazemi et al. first

Every direction below assumes we can find the refusal neuron ourselves, so this comes first. Apple has not released code, so we write our own (see [replication/README.md](replication/README.md)).

- **Models:** Qwen3-1.7B, Qwen3-8B, Llama-3.1-8B-Instruct. That covers both families on panther-sized GPUs; add 14B and 32B later.
- **Success check:** recover their neurons, Qwen3-8B L14:F7924 and Llama-3.1-8B L11:F4258, and roughly their JailbreakBench ASR (around 90%+ per model; exact numbers are in their Table 8).

## The five directions

### 1. Same neuron across languages and modalities? (Nadir's main ask → Papers A and B)
Find the refusal neuron in, for example, Qwen3-8B using English prompts. Then ask:
- **Does it fire** when the *same* harmful request is given in Arabic, Hindi, Swahili, as an image of the text, or as speech?
- **Does it fire at the same token**, the chat-template token just before the assistant answers?
- **Does suppressing it still jailbreak** the model in those languages and modalities?
- **If we run the neuron search in Arabic** (or on images), do we find the *same* neuron or a different one?

For modalities, use models built on the same LLM backbone, such as Qwen3 and a Qwen3-based VL or audio model. Then "the same neuron" literally means the same layer and index. Check each variant's backbone first.

### 2. Why translation and image jailbreaks work (→ Papers A and B)
Harmful requests in low-resource languages or embedded in images often get past safety. Hypothesis: the model *understands* the harm, so the harmful-concept neurons fire, but the refusal gate neuron stays quiet. That is "recognized but not refused", which we can measure directly per language and modality. A related result is AEGIS's "risk-to-refusal gap" in audio models.

### 3. The neuron as a cheap guard (→ Papers A and B)
Kazemi et al. show that one neuron's activation detects harmful prompts about as well as Llama-Guard-3-8B (AUROC 0.97 on XSTest for Llama-3.1-8B). Does that hold per language and per modality? If it does, it is a nearly free multilingual and multimodal safety detector.

### 4. Bridge to our EMNLP paper (→ Paper C)
Kazemi et al. test only *safety* refusals.
- Does suppressing their neuron also remove the **non-safety refusals**: CoCoNot humanizing, incomplete, indeterminate and unsupported? In our paper these survived safety-direction ablation at 0.3–0.5.
- Do our **11 category refusal directions** line up with that neuron's write vector (its `down_proj` column)? Kazemi et al. Table 7 runs the same check for Arditi's direction.
- Hypothesis linking the two papers: one gate neuron controls *whether* the model refuses, and category-specific directions or neurons shape *how* it refuses.

### 5. Testing Goodfire's essay on safety neurons (→ Paper C)
- **Neurons vs SAEs:** is our SAE "shared refusal core" (GemmaScope, Gemma-2-9B) mostly a few neurons in disguise? Run the neuron finder on Gemma-2 and compare it with the SAE latents. Use GemmaScope transcoders for an MLP-to-MLP comparison. Goodfire claims 88% of SAE activity is near-copies of single neurons.
- **Watch the neuron form:** use OLMo-2's open pretraining, SFT and DPO checkpoints to see when the refusal neuron appears, and whether it grows out of a related concept (for example "restricted or explicit content") as Goodfire's "recursive generalization" predicts.
- **Check the SwiGLU double-negative regime** in Kazemi et al.'s pole profiles, so that echo activations are not read as meaning.

## Shared resource worth building early
One **parallel benchmark** with the same harmful and benign prompts in N languages × {text, image-of-text, speech (TTS)}. Papers A and B both use it, and it could be released on its own.

## Open questions to settle with Nadir
- Which languages: Arabic and its dialects first, plus a high/low-resource spread?
- Which multimodal models share a backbone with the text models we replicate?
- Judge: Llama-Guard-3 plus an LLM judge as in Kazemi et al., and WildGuard for consistency with our EMNLP paper?
