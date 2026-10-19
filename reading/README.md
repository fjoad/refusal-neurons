# Reading list

Run `./download.sh` to fetch everything into `pdfs/` and `web/`. Both folders are gitignored. Entries are listed in [sources.tsv](sources.tsv).

## Core
- **Kazemi, Chegini, Safi (2026).** [A Single Neuron Is Sufficient to Bypass Safety Alignment in LLMs](https://arxiv.org/abs/2605.08513). The paper we build on. One MLP refusal neuron per model, found with a gradient × activation score. Suppressing it gives about 92% ASR on JailbreakBench across Qwen3 (1.7B–32B) and Llama-3.1 (8B, 70B). The neuron is already present in base models and works as a detector about as well as Llama-Guard. "Suicide" concept neurons are also shown. English and text only; no code released. ([Apple page](https://machinelearning.apple.com/research/single-neuron-safety-alignment))
- **Cammarata / Goodfire (2026).** [Neural Networks Are Already Clean](https://already-clean.goodfire.com/). Essay: polysemanticity came mostly from weak tools, old architectures and small models, not superposition. Neurons grow by "recursive generalization". About 90% of Qwen3-32B neuron poles have one job. Negative poles act as gates. Flags the SwiGLU double-negative regime. Argues for exploratory, neuron-first interpretability over SAEs.
- **Joad et al. (2026), our EMNLP paper.** [There Is More to Refusal in LLMs than a Single Direction](https://arxiv.org/abs/2602.02132). 11 category refusal directions are geometrically distinct, yet steering acts as one knob for *whether* the model refuses; the directions change *how*. SAEs show a shared core plus style-specific latents.
- **Arditi et al. (2024).** [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717). The original refusal direction.

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
