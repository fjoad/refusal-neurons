# Reading list

Run `./download.sh` to fetch everything into `pdfs/` and `web/`. Both folders are gitignored. Entries are listed in [sources.tsv](sources.tsv).

## Core
- **Kazemi, Chegini, Safi (2026).** [A Single Neuron Is Sufficient to Bypass Safety Alignment in LLMs](https://arxiv.org/abs/2605.08513). The paper we build on. One MLP refusal neuron per model, found with a gradient × activation score. Suppressing it gives about 92% ASR on JailbreakBench across Qwen3 (1.7B–32B) and Llama-3.1 (8B, 70B). The neuron is already present in base models and works as a detector about as well as Llama-Guard. "Suicide" concept neurons are also shown. English and text only; no code released. ([Apple page](https://machinelearning.apple.com/research/single-neuron-safety-alignment))
- **Cammarata / Goodfire (2026).** [Neural Networks Are Already Clean](https://already-clean.goodfire.com/). Essay: polysemanticity came mostly from weak tools, old architectures and small models, not superposition. Neurons grow by "recursive generalization". About 90% of Qwen3-32B neuron poles have one job. Negative poles act as gates. Flags the SwiGLU double-negative regime. Argues for exploratory, neuron-first interpretability over SAEs.
- **Joad et al. (2026), our EMNLP paper.** [There Is More to Refusal in LLMs than a Single Direction](https://arxiv.org/abs/2602.02132). 11 category refusal directions are geometrically distinct, yet steering acts as one knob for *whether* the model refuses; the directions change *how*. SAEs show a shared core plus style-specific latents.
- **Arditi et al. (2024).** [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717). The original refusal direction. Shorter [blog post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction); [code and prompt sets](https://github.com/andyrdt/refusal_direction).
- **Zhang, Zandsalimy, Sushmita (2026).** [Which Words a Model Fears: From a Refusal Direction to a Vocabulary-Wide Fear Lexicon](https://arxiv.org/abs/2611.02348). Harm words shift the hidden state along one shared direction whether or not they carry the request's meaning; the direction ranks the whole vocabulary from "feared" to "courage" words, and appending a feared word to a harmless prompt can make a model refuse. Eight models. Relevant to our harmful-word checks (XSTest, the planned benign-word prompt set). Shared by Majd, 2026-11-18.

## Multilingual safety
- **Wang et al. (NeurIPS 2025).** [Refusal Direction is Universal Across Safety-Aligned Languages](https://arxiv.org/abs/2505.17306). An English refusal direction transfers to 14 languages.
- **(2026)** [Who Transfers Safety? Cross-Lingual Shared Safety Neurons](https://arxiv.org/abs/2602.01283). A small set of shared safety neurons controls safety across languages.
- **(2026)** [When Safety Speaks a Language: Safety–Language Identity Entanglement](https://arxiv.org/abs/2608.29936). Universality breaks down when safety features are entangled with language identity.

## Multimodal safety
- **(2026)** [Targeted Interpretable Safety Neuron Enhancement for Multilingual VLMs](https://arxiv.org/abs/2604.08881). Safety neurons in multilingual VLMs; intervenes on about 3% of neurons.
- **(2026)** [When Safety Overrides Vision](https://arxiv.org/abs/2608.18628). VLMs refuse even when they perceive the image; suppressing refusal activations restores answers.
- **(2026)** [Jailbreaking VLMs Through the Visual Modality](https://arxiv.org/abs/2605.00583). Harmful intent delivered through images bypasses text-trained safety.
- **(2026)** [AEGIS: Audio LM Jailbreaks](https://arxiv.org/abs/2609.29287). Audio jailbreaks show a "risk-to-refusal gap": risk is decodable in middle layers but doesn't trigger refusal.
- **Gupta et al. (2025).** ["I am bad": Interpreting Audio Jailbreaks](https://arxiv.org/abs/2502.00718). Universal adversarial audio works by embedding linguistic features in the signal.
- **Foo and Lee (2026).** [Causal Localization of the Refusal Direction in Audio Language Models](https://arxiv.org/abs/2609.22260). Fits a harmful-against-benign direction at the audio-to-LM interface and at layers of the text LM, removes it, and measures the change in refusal. Across five audio LMs, Qwen2.5-Omni among them, the effect sits in a mid-to-late band of the text LM; the interface directions do little. The closest prior work to our speech part: direction level only, no neurons. Shared by Nadir, 2026-11-19.

## Multimodal: models, tools and datasets (listed by Nadir, 2026-11-18)
- **Omni models** (text, speech and images in one network): [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it) (gated), [LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B), [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct). "Qwen3.8-Omni-Flash" was not on Hugging Face under that name on 2026-11-18; the open Qwen omni models are [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct), [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) and [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B).
- **In use for the first speech runs (Asim, from 2026-11-18):** [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B). Only its "thinker" is studied: the part that reads text or audio and writes the text answer. The "talker" only turns that text into speech.
- **Text-to-speech:** [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2), 17 languages including Arabic, to turn our text prompts into speech.
- **Input handling:** the [Qwen2.5-Omni processor in transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py) shows which token positions belong to which modality.
- **[FBK-MT/RedVox](https://huggingface.co/datasets/FBK-MT/RedVox)** (gated): spoken safety prompts in German, English, Spanish, French and Italian. Paper: [RedVox: Safety and Fairness Gaps in Speech Models Across Languages](https://arxiv.org/abs/2606.26968).
- **[Leyiii/Omni-SafetyBench](https://huggingface.co/datasets/Leyiii/Omni-SafetyBench)**: 972 harmful requests, each in 24 forms (text, image, video, audio and combinations), over 23,000 samples, English.

## Safety neurons (neuron-level prior work)
- **Chen et al. (2024).** [Towards Understanding Safety Alignment: Safety Neurons](https://arxiv.org/abs/2406.14144). About 5% of neurons account for 90%+ of safety behavior; they already exist in base models.
- **Wu et al. (2025).** [NeuroStrike](https://arxiv.org/abs/2509.11864). Pruning under 0.6% of neurons gives 76.9% ASR, with transfer to multimodal variants.
- **Wang et al. (2026).** [SafeNeuron](https://arxiv.org/abs/2602.12158). Freezes safety neurons during fine-tuning to spread safety more broadly (a defense).

## Neurons vs SAEs
- **Arora et al. (ICML 2026).** [Language Model Circuits Are Sparse in the Neuron Basis](https://arxiv.org/abs/2601.22594). MLP neurons are as sparse a basis as SAEs for circuits.
- **Kantamneni, Engels, Rajamanoharan, Tegmark, Nanda (2025).** [Are Sparse Autoencoders Useful? A Case Study in Sparse Probing](https://arxiv.org/abs/2502.16681). SAE probes don't reliably beat baselines.
- **Smith, Rajamanoharan, …, Nanda (GDM, 2025).** [Negative Results for SAEs on Downstream Tasks and Deprioritising SAE Research](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks). Linear probes beat SAEs at out-of-distribution harmful-intent detection; the team deprioritized SAE research. Companion post: [Takeaways from SAE probing](https://www.alignmentforum.org/posts/osNKnwiJWHxDYvQTD/takeaways-from-our-recent-work-on-sae-probing).
- **Wu et al. (2025).** [AxBench](https://arxiv.org/abs/2501.17148). Simple baselines beat SAEs at steering.
- **(2026)** [Steering LLMs? Actually, SAEs Can Outperform Simple Baselines](https://arxiv.org/abs/2605.31183). Rebuttal to AxBench (preprint).
- **Cunningham et al. (2023).** [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600). One of the two papers that introduced SAEs for LLM interpretability.
- **Bricken et al. (2023).** [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features). Anthropic's SAE paper, the other origin.
- **Templeton et al. (2024).** [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity). SAEs on Claude 3 Sonnet; safety-relevant features.
- **Lieberum et al. (2024).** [Gemma Scope](https://arxiv.org/abs/2408.05147). The SAEs we used in the EMNLP paper.
