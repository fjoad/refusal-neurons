# Gatekeeper

**Is safety refusal gated by the same single neuron across languages and modalities?**

Kazemi et al. (2026, [arXiv:2605.08513](https://arxiv.org/abs/2605.08513)) show that one MLP neuron in an LLM acts as a refusal "gate". Pinning that neuron to a constant bypasses safety on about 92% of JailbreakBench prompts, across 7 models from 1.7B to 70B. Their study is English-only and text-only.

This project asks whether that gate is universal. Does the same neuron fire, and can it be suppressed, when the harmful request comes:
- in another language (Arabic, low-resource languages, code-switching)?
- as an image or as speech?

It also asks how the neuron view relates to residual-stream directions and SAE features. That second question builds on our EMNLP 2026 paper ([arXiv:2602.02132](https://arxiv.org/abs/2602.02132)) and on Goodfire's "[Neural Networks Are Already Clean](https://already-clean.goodfire.com/)".

QCRI / HBKU: Faaiz Joad, Nadir Durrani, and collaborators. Started October 2026.

## Plan

One project folder and shared experiment code, written up as **2–3 separate papers**. See [RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md) for the full list of directions.

| Step | What | Output |
|---|---|---|
| 0 | Replicate Kazemi et al. on small Qwen3 and Llama-3.1 models | Our own refusal-neuron pipeline ([replication/](replication/)) |
| Paper A | The same neuron across **languages** (directions 1–3, multilingual) | Multilingual paper |
| Paper B | The same neuron across **modalities**, image and speech (directions 1–3, multimodal) | Multimodal paper; could merge with A if results are small |
| Paper C | **Neurons vs directions vs SAEs** (directions 4–5) | Interpretability paper bridging our EMNLP paper, Kazemi et al., and Goodfire |

## Layout

```
README.md               this file
RESEARCH_DIRECTIONS.md  the five open directions and how they map to papers
docs/glossary.md        plain-language explanations of neurons, poles, superposition, etc.
reading/                reading list (README.md), sources.tsv, download.sh
  pdfs/, web/           downloaded copies (gitignored; run reading/download.sh)
replication/            step 0: reproducing the single-neuron paper
```

## Getting the reading material

```bash
./reading/download.sh
```

The PDFs and web pages are third-party copyrighted material. They are not committed; only the links are.
