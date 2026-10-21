# Refusal Neurons

**Is safety refusal gated by the same single neuron across languages and modalities?**

Kazemi et al. (2026, [arXiv:2605.08513](https://arxiv.org/abs/2605.08513)) show that one MLP neuron in an LLM acts as a refusal "gate". Pinning that neuron to a constant bypasses safety on about 92% of JailbreakBench prompts, across 7 models from 1.7B to 70B. Their study is English-only and text-only.

This project asks whether that gate is universal. Does the same neuron fire, and can it be suppressed, when the harmful request comes:
- in another language (Arabic, low-resource languages, code-switching)?
- as an image or as speech?

It also asks how the neuron view relates to residual-stream directions and SAE features. That second question builds on our EMNLP 2026 paper ([arXiv:2602.02132](https://arxiv.org/abs/2602.02132)) and on Goodfire's "[Neural Networks Are Already Clean](https://already-clean.goodfire.com/)".

QCRI / HBKU: Faaiz Joad, Nadir Durrani, and collaborators. Started October 2026.

> **Agents and new collaborators:** this README is the project guide. Read it top to bottom, then [AGENTS.md](AGENTS.md) for the operational rules.

## Roadmap

The work runs in this order; each step gates the next. Keep the state column current as work lands. Results with numbers go in the **results log** at the bottom of [replication/README.md](replication/README.md).

| Step | What it means | State |
|---|---|---|
| Setup | `refneu` env, models and data splits on panther | ✅ |
| **M1**: check published neurons | Apple's neuron IDs separate harmful from harmless prompts, and pinning them stops refusal | ✅ Qwen3-1.7B, Qwen3-8B, Llama-3.1-8B |
| Arabic teaser | The English-found neuron on parallel Arabic prompts | ✅ transfers (AUROC 0.954 vs 0.977 in English) |
| **M2**: rediscover the neurons | Our gradient × activation search should find L13:F3270 (Qwen3-1.7B) and L14:F7924 (Qwen3-8B) by itself, at least in its top 5, and then pick the right one by reranking on HarmBench ASR with an *m* sweep | 🟡 search code written and reviewed; acceptance run and rerank step pending |
| Judge | One judge used for every language (see Open questions) | ⬜ |
| **M3**: full evaluation | JailbreakBench ASR, anchor variant, MMLU/GSM8K cost, XSTest detection AUROC; English first | ⬜ |
| Arabic search | Run the search on Arabic splits: same neuron or a different one? | ⬜ |
| Modalities | Image and speech, using models that share a backbone with our text models | ⬜ |

## Decisions so far, and why

These were settled in discussion. Don't reopen them without a new reason.

- **Replicate first, extend second.** Every direction depends on finding the neuron ourselves. Apple released no code, so we reimplement from the paper (Section 2 and Appendix A).
- **Models:** Qwen3-1.7B, Qwen3-8B, Llama-3.1-8B-Instruct. They cover both families in the paper and fit on one V100-32GB. Larger models come later, if needed.
- **fp16 on V100.** V100s have no native bf16. An fp16-vs-fp32 check on Qwen3-1.7B gave identical results.
- **Data follows Arditi et al. exactly**, as Apple did: 128 harmful and 128 harmless train prompts, HarmBench validation (191 after removing JailbreakBench overlaps, which matches the paper's count), and JailbreakBench as the test set.
- **Arabic only for now.** If we get a full replication plus an Arabic contribution, that is enough for Paper A even if other languages don't work. Modalities come after Arabic. Add languages along a resource spectrum later (for example French or Chinese, then Swahili or Bengali).
- **Human-translated multilingual data, not our own machine translation**, so reviewers can't blame the translations. That means MultiJail (MIT) for harmful prompts and the Aya dataset (Apache-2.0) for harmless ones.
- **The search is the main tool of the project**, not just a replication step. "Same neuron in Arabic?" is only answered honestly by running the search in Arabic. Checking whether the English neuron fires in Arabic is the weaker test.
- **One judge for every language.** Llama-Guard-3, which Apple used, likely doesn't cover Arabic. Comparing English numbers from one judge with Arabic numbers from another would not be valid.
- **Refusal is scored automatically and responses stay out of logs.** Generated text goes to the output JSON only. Read it with `replication/show_responses.py`.
- **Several papers from one codebase** (below), not everything in one paper.

## Open questions
- **Judge:** an LLM judge (Claude, as Apple used, which reads Arabic) vs translating responses back to English and using Llama-Guard-3 or WildGuard (WildGuard matches our EMNLP paper). Plan: run two on a sample, check agreement, then pick.
- **Multimodal models** that share a backbone with our text models (for example a Qwen3-based VL or audio model). Check each model's backbone before using it.
- **MultiJail's baseline refusal rate is low** (~0.45 by string match). Some prompts are borderline. A proper judge will tell us whether we need a stricter harmful set.

## Plan: three papers, one codebase

See [RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md) for the five directions in detail.

| Step | What | Output |
|---|---|---|
| 0 | Replicate Kazemi et al. on small Qwen3 and Llama-3.1 models | Our own refusal-neuron pipeline ([replication/](replication/)) |
| Paper A | The same neuron across **languages** (directions 1–3, multilingual) | Multilingual paper |
| Paper B | The same neuron across **modalities**, image and speech (directions 1–3, multimodal) | Multimodal paper; could merge with A if results are small |
| Paper C | **Neurons vs directions vs SAEs** (directions 4–5) | Interpretability paper bridging our EMNLP paper, Kazemi et al., and Goodfire |

A parallel benchmark (the same prompts in N languages × text, image of text, speech) would serve Papers A and B and could be released on its own.

## Background in one paragraph

Our EMNLP paper found refusal in the **residual stream** as several category-specific directions that share a core. Steering along any of them acts as one knob for *whether* the model refuses, and the directions change *how* it refuses. Kazemi et al. work with raw **MLP neurons**, which have a privileged basis, and find one neuron is sufficient for *whether*. Goodfire argues that raw neurons are already interpretable in general, so SAEs are often unnecessary. A working hypothesis linking them: one gate neuron decides whether to refuse, and category-specific directions shape how. Terms such as poles, privileged basis, gate × value, and superposition are explained simply in [docs/glossary.md](docs/glossary.md).

## Running experiments

Experiments run on **panther** (QCRI SLURM). The repo is cloned at `~/refusal-neurons` there, and jobs write to `logs/` and `outputs/`, which are gitignored. See [AGENTS.md](AGENTS.md) for the edit → push → pull → `sbatch` loop and the panther gotchas.

| Script | What it does |
|---|---|
| `env/setup_env.sh` | SLURM job that builds the `refneu` conda env (torch 2.5.1 + `env/requirements.txt`) |
| `scripts/download_models.sh` | SLURM job that downloads Hugging Face models; the login node kills large downloads |
| `data/prepare_data.py` | Builds the Arditi/Apple splits in `data/splits/` |
| `data/prepare_arabic.py` | Builds parallel MultiJail en/ar (100 prompts) and Aya harmless en/ar splits |
| `replication/neuron_check.py` (+ `.sh`) | M1: given a known (layer, neuron), measures separation (AUROC, per-token activations) and refusal rate when pinning it to several values. `--harmful/--harmless` choose splits, `--tag` labels the output |
| `replication/neuron_search.py` (+ `.sh`) | M2: Apple's gradient × activation search over the first 2/3 of layers; prints and saves the top-5 candidates |
| `replication/show_responses.py` | Prints or writes a few generated responses from an output JSON, for a human to read |

Examples:
```bash
sbatch replication/neuron_check.sh Qwen/Qwen3-8B 14 7924 float16
sbatch replication/neuron_check.sh Qwen/Qwen3-8B 14 7924 float16 --harmful multijail_ar --harmless aya_harmless_ar --tag mj_ar
python replication/show_responses.py outputs/neuron_check/<file>.json --out /tmp/responses.txt
```

## Layout

```
README.md               this file: the project guide, decisions and roadmap
AGENTS.md               rules for AI agents (CLAUDE.md just imports it)
RESEARCH_DIRECTIONS.md  the five open directions and how they map to papers
docs/glossary.md        plain-language explanations of neurons, poles, superposition, etc.
reading/                reading list (README.md), sources.tsv, download.sh
  pdfs/, web/           downloaded copies (gitignored; run reading/download.sh)
env/                    conda env job and requirements
scripts/                model download job
data/                   split builders and the splits themselves (data/splits/*.json)
replication/            step 0: method notes, scripts, and the results log (README.md)
```

## Getting the reading material

```bash
./reading/download.sh
```

The PDFs and web pages are third-party copyrighted material. They are not committed; only the links are.
