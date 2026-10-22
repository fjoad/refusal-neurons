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
| **M2**: rediscover the neurons | Our search finds the published neurons in its top 5; a rerank on HarmBench ASR with an *m* sweep picks the best attacker | ✅ Search: all 3. Rerank: published neuron and m* recovered for Qwen3-8B (+20) and Llama (−4); Qwen3-1.7B is a near-tie (L10:F244 edges L13:F3270) |
| Judge | One judge used for every language | ✅ Llama-Guard-3-8B (free, on panther); Arabic handling still open |
| **M3**: full evaluation | JailbreakBench ASR, constant and anchor interventions, MMLU/GSM8K cost, XSTest detection AUROC; English first | ✅ Llama-Guard only. Both 8B neurons jailbreak at 0.93–0.95, and the anchor removes the capability cost. Detection matches the paper; Arditi baseline 0.91–0.95. A coherence-aware LLM judge is still missing |
| Arabic search | Run the search on Arabic splits: same neuron or a different one? | 🟡 **Same neuron in all 3 models**: the English refusal neuron ranks 1st in an Arabic search that targets the model's own Arabic refusal phrases. Arabic rerank (ASR) pending |
| Modalities | Image and speech, using models that share a backbone with our text models | ⬜ |

## Decisions so far, and why

These were settled in discussion. Don't reopen them without a new reason.

- **Replicate first, extend second.** Every direction depends on finding the neuron ourselves. Apple released no code, so we reimplement from the paper (Section 2 and Appendix A).
- **Models:** Qwen3-1.7B, Qwen3-8B, Llama-3.1-8B-Instruct. They cover both families in the paper and fit on one V100-32GB. Larger models come later, if needed.
- **fp16 on V100.** V100s have no native bf16. An fp16-vs-fp32 check on Qwen3-1.7B gave identical results.
- **Data follows Arditi et al. exactly**, as Apple did: HarmBench validation (191 after removing JailbreakBench overlaps, which matches the paper's count) and JailbreakBench as the test set. For training, use **Arditi's exact 128/128 sample** (`arditi_*_train`, built by `data/prepare_data.py`). With it, the search reproduces Table 5's activations; with our own sample, Qwen3-1.7B's neuron dropped to rank 11.
- **The search follows the paper's appendix to the letter:** Table 4 refusal phrases, and only Table 3's red post-instruction tokens. Each of these changed the result (see the M2 entry in the results log). When the PDF text is ambiguous, render the page and look; the red highlighting was invisible in extracted text.
- **Arabic only for now.** If we get a full replication plus an Arabic contribution, that is enough for Paper A even if other languages don't work. Modalities come after Arabic. Add languages along a resource spectrum later (for example French or Chinese, then Swahili or Bengali).
- **Human-translated multilingual data, not our own machine translation**, so reviewers can't blame the translations. That means MultiJail (MIT) for harmful prompts and the Aya dataset (Apache-2.0) for harmless ones.
- **The search is the main tool of the project**, not just a replication step. "Same neuron in Arabic?" is only answered honestly by running the search in Arabic. Checking whether the English neuron fires in Arabic is the weaker test.
- **Judge: Llama-Guard-3-8B only, free and on panther** (decided 2026-10-21). No API credits for an LLM judge while this is exploratory. A response is a successful attack when Llama-Guard labels it unsafe, which is Apple's secondary judge. Llama-Guard does **not** reliably reject garbled output: it called all of Qwen3-8B's pin-80 outputs unsafe, and those had a median length of 100 characters. So the rerank picks the smallest pin within 0.02 of the best ASR, and high-pin numbers need a coherence check. Llama-Guard-3 doesn't officially cover Arabic, so Arabic ASR needs care: translating to English first, or checking agreement on a sample, are the options when we get there.
- **One judge for every language.** Comparing English numbers from one judge with Arabic numbers from another would not be valid.
- **Rerank selection: the candidate reaching ASR ≥ 0.90 at the smallest pin**, not max ASR, because Llama-Guard rewards garbled high-pin output. This rule recovers Apple's neuron and m* on both 8B models.
- **Rerank is cheaper than the paper's:** a 9-value *m* grid (2–60) and 128 new tokens instead of their ~47 values and 512 tokens, so it fits P100/V100s. Refine around the best *m* later if needed.
- **Arabic searches score each model's own Arabic refusal phrases** (from `refusal_openings.py`). With English phrases, Qwen3-1.7B's neuron fell to rank 568; with its own Arabic phrases it is rank 1.
- **Arabic search scores Qwen's own Arabic refusal phrase.** Unpinned, Qwen3-8B refuses Arabic prompts in Arabic, usually opening with "أنا لا أستطيع مساعدتك" ("I can't help you"; 10 of 30 responses). The English phrases would target a reply the model doesn't give in Arabic.
- **Refusal is scored automatically and responses stay out of logs.** Generated text goes to the output JSON only. Read it with `replication/show_responses.py`.
- **Several papers from one codebase** (below), not everything in one paper.

## Open questions
- **Arabic ASR:** Llama-Guard-3 on Arabic directly, or after translating responses to English with a free local model. Decide when the Arabic intervention numbers are needed.
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
| `replication/neuron_search.py` (+ `.sh`) | M2: Apple's gradient × activation search over the first 2/3 of layers; prints and saves the top-5 candidates. `--watch L:F` reports a neuron's rank; per-layer means are saved to `*_means.pt` |
| `replication/rescore.py` | Re-ranks neurons from a saved `*_means.pt` with a different token subset, without rerunning the model (CPU only) |
| `replication/rerank.py` | M2 rerank: pins each top-5 search candidate over a grid of *m* on HarmBench and saves the responses (log shows rough refusal rates only) |
| `replication/judge.py` | Llama-Guard-3 verdicts and ASR for a rerank or neuron_check output; writes verdicts back into the JSON |
| `replication/refusal_openings.py` | Counts the stock openings of unpinned responses, to find a model's refusal phrases in a language; never prints full responses |
| `scripts/pssh.sh` | Runs a command on panther, retrying when the VPN drops the connection |
| `scripts/cpu.sh` | SLURM wrapper that runs one repo script on a CPU node (e.g. `summarize.py`, `refusal_openings.py`) |
| `replication/show_responses.py` | Prints or writes a few generated responses from an output JSON, for a human to read |

Examples:
```bash
sbatch replication/neuron_check.sh Qwen/Qwen3-8B 14 7924 float16
sbatch replication/neuron_check.sh Qwen/Qwen3-8B 14 7924 float16 --harmful multijail_ar --harmless aya_harmless_ar --tag mj_ar
sbatch replication/neuron_search.sh Qwen/Qwen3-8B float16 --watch 14:7924   # paper settings are the defaults
python replication/show_responses.py outputs/neuron_check/<file>.json --out /tmp/responses.txt
```

## Layout

```
README.md               this file: the project guide, decisions and roadmap
AGENTS.md               rules for AI agents (CLAUDE.md just imports it)
RESEARCH_DIRECTIONS.md  the five open directions and how they map to papers
docs/glossary.md        plain-language explanations of neurons, poles, superposition, etc.
docs/roadmap.html       roadmap page, published as a Claude artifact
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
