# Research directions

Written down on 2026-10-19 after a meeting with Nadir Durrani. Nadir's framing: build on Kazemi et al. (2026) for **multilingual and multimodal safety refusal**.

The terms used here (neuron, pole, gate, ASR) are explained in [docs/glossary.md](docs/glossary.md).

## Step 0: replicate Kazemi et al. first

Every direction below assumes we can find the refusal neuron ourselves, so this comes first. Apple has not released code, so we write our own (see [replication/README.md](replication/README.md)).

- **Models:** Qwen3-1.7B, Qwen3-8B, Llama-3.1-8B-Instruct. That covers both families on panther-sized GPUs; add 14B and 32B later.
- **Success check:** recover their neurons, Qwen3-8B L14:F7924 and Llama-3.1-8B L11:F4258, and roughly their JailbreakBench ASR (around 90%+ per model; exact numbers are in their Table 8).
- **Status (2026-10-27): done.** Same neuron and pin as Apple on both 8B models, 93–95% ASR; see the README's status tables.

## The five directions

### 1. Same neuron across languages and modalities? (Nadir's main ask → Papers A and B)
**Status:** Arabic started. The English neuron fires on Arabic prompts at the same token, and a search run on Arabic prompts gives largely the same top-5 neurons. Arabic judging is validated (Qwen3Guard and Llama-Guard on an NLLB translation agree, kappa 0.88), and Apple's Qwen3-8B neuron at +20 jailbreaks Arabic prompts (0.07 → 0.95). In progress: the full Arabic replication (rerank running). Not done: other languages, images, speech.

Find the refusal neuron in, for example, Qwen3-8B using English prompts. Then ask:
- **Does it fire** when the *same* harmful request is given in Arabic, Hindi, Swahili, as an image of the text, or as speech?
- **Does it fire at the same token**, the chat-template token just before the assistant answers?
- **Does suppressing it still jailbreak** the model in those languages and modalities?
- **If we run the neuron search in Arabic** (or on images), do we find the *same* neuron or a different one?

For modalities, use models built on the same LLM backbone, such as Qwen3 and a Qwen3-based VL or audio model. Then "the same neuron" literally means the same layer and index. Check each variant's backbone first.

### 2. Why translation and image jailbreaks work (→ Papers A and B)
**Status:** not started. The models refuse less in Arabic (string match: Qwen3-1.7B 0.40 → 0.20, Llama 0.77 → 0.58) while the neuron still fires, which is a first hint worth measuring properly.

Harmful requests in low-resource languages or embedded in images often get past safety. Hypothesis: the model *understands* the harm, so the harmful-concept neurons fire, but the refusal gate neuron stays quiet. That is "recognized but not refused", which we can measure directly per language and modality. A related result is AEGIS's "risk-to-refusal gap" in audio models.

### 3. The neuron as a cheap guard (→ Papers A and B)
**Status:** first result. On Qwen3-8B the neuron detects Arabic harmful prompts better than Llama-Guard-3 (AUROC 0.956 vs 0.910); a cut-off set in English still works in Arabic (95.5% → 88.5% accuracy).

Kazemi et al. show that one neuron's activation detects harmful prompts about as well as Llama-Guard-3-8B (AUROC 0.97 on XSTest for Llama-3.1-8B). Does that hold per language and per modality? If it does, it is a nearly free multilingual and multimodal safety detector.

### 4. Bridge to our EMNLP paper (→ Paper C)
**Status:** first result. In all three models the refusal neuron is the neuron most aligned with Arditi's direction in its own layer (Experiment 7). Our 11 category directions are not compared yet.

Kazemi et al. test only *safety* refusals.
- Does suppressing their neuron also remove the **non-safety refusals**: CoCoNot humanizing, incomplete, indeterminate and unsupported? In our paper these survived safety-direction ablation at 0.3–0.5.
- Do our **11 category refusal directions** line up with that neuron's write vector (its `down_proj` column)? Kazemi et al. Table 7 runs the same check for Arditi's direction.
- Hypothesis linking the two papers: one gate neuron controls *whether* the model refuses, and category-specific directions or neurons shape *how* it refuses.

### 5. Testing Goodfire's essay on safety neurons (→ Paper C)
**Status:** not started.

- **Neurons vs SAEs:** is our SAE "shared refusal core" (GemmaScope, Gemma-2-9B) mostly a few neurons in disguise? Run the neuron finder on Gemma-2 and compare it with the SAE latents. Use GemmaScope transcoders for an MLP-to-MLP comparison. Goodfire claims 88% of SAE activity is near-copies of single neurons.
- **Watch the neuron form:** use OLMo-2's open pretraining, SFT and DPO checkpoints to see when the refusal neuron appears, and whether it grows out of a related concept (for example "restricted or explicit content") as Goodfire's "recursive generalization" predicts.
- **Check the SwiGLU double-negative regime** in Kazemi et al.'s pole profiles, so that echo activations are not read as meaning.

## Ideas added at the 22 October meeting

- **The refusal direction inside the MLP.** Compute Arditi's mean-difference direction over the MLP's neuron values instead of the residual stream, and steer or ablate with it there. Decompose it: is the refusal neuron its most-weighted entry, and how much of the direction does it carry? When Arditi's direction works, is this neuron the only one that matters? First check: Apple's appendices don't build an MLP-space direction (Appendix G works from the residual direction). Bridges directions 1 and 4.
- **The full method in Arabic.** Find the neuron in Arabic with the same algorithm (search, rerank, attack) and compare its effectiveness with the English neuron.
- **A language-agnostic refusal neuron as one guard across languages** (directions 1 and 3). Search in every language of PolyRefuse and MultiJail, intersect the top-5 neurons, and test the most-shared neuron as a detector everywhere. If no single neuron works, look for one per language family or script, then one per language.

## Ideas added at the 17 November meeting (for a later paper)
- **Single neurons for other behaviours.** Apple's search finds one neuron that gates refusal. Run the same search with a different target behaviour and see whether one neuron controls it.
- **A culture neuron.** With a culture dataset as the contrast, look for a neuron that shifts the model's cultural alignment when pinned (the meeting's example: a Muslim-aligned Qwen from one neuron). Needs a dataset choice first.

## Shared resource worth building early
One **parallel benchmark** with the same harmful and benign prompts in N languages × {text, image-of-text, speech (TTS)}. Papers A and B both use it, and it could be released on its own.

## Open questions to settle with Nadir
- Which languages: Arabic and its dialects first, plus a high/low-resource spread?
- Which multimodal models share a backbone with the text models we replicate?
- Judge: Llama-Guard-3 plus an LLM judge as in Kazemi et al., and WildGuard for consistency with our EMNLP paper?

## Added 2026-11-24: concept induction across languages and modalities (Nadir)

Apple shows concept neurons only for suicide-related content. Pass matched inputs (the same content as text and as speech, in several languages) through the omni model's thinker and ask: do the same neurons respond to a concept such as suicide, or another safety topic, across languages and modalities? Then intervene on those neurons and test whether the concept becomes more likely to appear in the model's answers. Not started. It reuses the search's machinery with a concept as the target in place of refusal.

## Added 2026-11-26: one neuron reads the request, others decide (hypothesis)

**What we saw.** On Qwen2.5-Omni-7B, when L13:F4787 is pinned, some harmful requests get neither a refusal nor an answer. The model replies that it is not sure what is being asked. The judge counts these as refusals, which is why MultiJail looked weak (65–75%) next to PolyRefuse (81–98%).

**The evidence so far** (English, counted from the answers' first words; results log, 26 November):

| Intervention | "Not sure what you are asking" replies |
|---|---|
| Nothing changed | 0 to 2 of 100 |
| L13:F4787 pinned at +30, PolyRefuse | 13 of 100 |
| L13:F4787 pinned at +30, MultiJail | 27 of 100 |
| L14:F13014 pinned at +40, the same PolyRefuse prompts | 1 of 100 |
| Nanda's direction removed, the same PolyRefuse prompts | 0 of 100 |
| L13:F4787 at +2 to +6 / +10 / +20 / +30 (HarmBench, 191) | 0 / 2 / 12 / 21 |
| L16:F1870 at any pin (HarmBench, 191) | 0 |

The prompts concerned are the shorter ones, and the same prompts in every language.

**The hypothesis.** The refusal neurons do not all do the same job. L13:F4787 carries part of the model's reading of the request ("this asks for something harmful"), and L14:F13014 and L16:F1870 sit on the decision side ("so refuse"). Pin a decision neuron and the model still knows what was asked and answers it. Pin the reading neuron far to the other side and, for a request whose harmful act is most of its content, the model no longer has a request to answer. This is the "is *this is harmful* separate from *how the model responds*?" question from the 17 November meeting, now with a behaviour that tells the two apart.

**What would support it, and what would not.**

| Test | If the hypothesis holds | State |
|---|---|---|
| The same prompts with another neuron pinned, or the direction removed | They get answered | Seen on PolyRefuse English (1 and 0 of 100) |
| MultiJail with L14:F13014 pinned | Fewer "not sure" replies, and a score closer to PolyRefuse's | Queued |
| Short harmless prompts with L13:F4787 pinned | Few "not sure" replies: the effect is about harmful content, not short prompts in general. Many would mean the pin simply damages reading | Not run |
| Where the neurons fire | L13:F4787 fires on the harmful words of the request; the decision neurons only at the template tokens | Not run (the text models' token-level script can be reused) |
| Each neuron's share of Nanda's direction | The decision neurons write along it; L13:F4787 less so | Not run |
| A third judge label, "does not engage", in every language | The same pattern outside English | Not run; the current count only matches English phrases |
| Text models | Apple's neurons there may split the same way (Qwen3-8B's L14:F7924 against the other top-5 neurons) | Not run |

**Why it matters for the paper.** It gives a reason to prefer L14:F13014 as "the" neuron on this model (also the most shared across languages and the better guard). It says the attack numbers for L13:F4787 understate how often refusal itself is switched off. And it is a small, concrete version of Nadir's RQ3: different neurons contribute different things to one refusal behaviour.

**Not yet ruled out.** That a pin of +30 on L13:F4787 is simply too strong and breaks comprehension in general (the harmless-prompt test separates this); that "not sure what you are asking" is itself a soft refusal the model falls back to.

