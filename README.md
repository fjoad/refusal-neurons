# Refusal Neurons

**Is safety refusal gated by the same single neuron across languages and modalities?**

Kazemi et al. (2026, [arXiv:2605.08513](https://arxiv.org/abs/2605.08513)) show that one MLP neuron in an LLM acts as a refusal "gate". Pinning that neuron to a constant bypasses safety on about 92% of JailbreakBench prompts, across 7 models from 1.7B to 70B. Their study is English-only and text-only. We call it "the Apple paper".

This project asks whether that gate is universal. Does the same neuron fire, and can it be suppressed, when the harmful request comes:
- in another language (Arabic first; later low-resource languages and code-switching)?
- as an image or as speech?

It also asks how the neuron view relates to residual-stream directions and SAE features. That builds on our EMNLP 2026 paper ([arXiv:2602.02132](https://arxiv.org/abs/2602.02132)) and on Goodfire's "[Neural Networks Are Already Clean](https://already-clean.goodfire.com/)".

QCRI / HBKU: Faaiz Joad, Nadir Durrani, and collaborators. Started October 2026. Repo: github.com/fjoad/refusal-neurons (public).

## Start here

Read in this order:
1. **This README:** what the project is, where it stands, decisions, how to run things.
2. **[docs/explained_simply.md](docs/explained_simply.md):** every experiment and result in plain language. It's the fastest way to understand the work, and the basis for the slides.
3. **[AGENTS.md](AGENTS.md):** operational rules (panther, harmful content, keeping docs current). Required for AI agents.
4. **[replication/README.md](replication/README.md):** the method in detail, every deviation from the paper, and the dated **results log** with all numbers.
5. **[docs/glossary.md](docs/glossary.md):** plain-language definitions of every term used here.

## Where things stand (updated 2026-10-27)

**The replication of the Apple paper is done for three models**, scored with one judge (Llama-Guard-3). **The same replication in Arabic is in progress** (see "Work in progress" below).

**Target: ARR, 4 January 2027 (ACL 2027).** The topic is low-hanging fruit, so submit as soon as possible. Authors: Faaiz Joad and Nadir Durrani. Weekly meeting every Thursday (see "Meetings" below).

| Step | What it means | State |
|---|---|---|
| Setup | `refneu` env, models and data splits on panther | ✅ |
| **M1**: check published neurons | Apple's neuron IDs separate harmful from harmless prompts, and pinning them stops refusal | ✅ Qwen3-1.7B, Qwen3-8B, Llama-3.1-8B |
| **M2**: rediscover the neurons | Our search finds the published neurons in its top 5; the rerank (pin-value sweep) picks the best attacker | ✅ Search: all 3. Rerank: same neuron and same pin as Apple on both 8B models; Qwen3-1.7B is a near-tie |
| Judge | One judge for every language | ✅ Llama-Guard-3-8B (free). No coherence check yet |
| **M3**: full evaluation | JailbreakBench attack, constant and anchor pins, MMLU/GSM8K cost, XSTest detection | ✅ Matches the paper (Llama-Guard only) |
| Other paper experiments | Sections 3.1, 3.2, 3.3, 3.4 and Appendix G | ✅ except 3.3 on Llama (needs a coherence judge); 3.5 and 4 not done |
| **Arabic replication** | The whole Apple paper again in Arabic, with the Arabic neuron found from scratch, on two tracks (PolyRefuse, MultiJail) | 🟡 Experiment 1 search done (largely the same neurons as English). Rerank: Qwen3-1.7B judged (near-tie on PolyRefuse, under 0.90 on MultiJail); the 8B reranks still generating. Experiments 2–8 not started |
| Modalities | Image and speech | ⬜ |

### Work in progress

What is running or parked, where its output lands, and what comes next. Job logs are in `logs/` on panther; job scripts are in `scripts/jobs/`.

| Work | Script | Output (on panther) | Next step |
|---|---|---|---|
| Arabic Experiment 1: refusal sentences, search and rerank answers, both tracks, 3 models | `scripts/jobs/arabic_exp1.sh <model> <English L:F>` | `outputs/refusal_phrases/<model>_ar.json`, `outputs/neuron_search/<model>_float16_{poly,mj}_ar_full.json`, `outputs/rerank/<model>_float16_{poly,mj}_ar.json` | Judge each model's rerank files with `sbatch scripts/jobs/judge_ar_rerank.sh <model>` (both Arabic judges, both tracks), then pick each Arabic neuron and pin by the 0.90 rule. Qwen3-1.7B is judged (job 436940): the rule doesn't settle a winner (see the results log); Qwen3-8B and Llama are being judged (437021, 437022); their judging is queued to start when each ends (437021/437022) |
| Parked: the refusal direction inside the MLP | `replication/mlp_direction.py` (written, **not reviewed or run**) | `outputs/mlp_direction/`, `outputs/evaluate/<model>_L<L>_mlpdir_<tag>.json` | Review it line by line, then run after the Arabic replication |
| Parked: refusal phrases for all 24 language sets | `scripts/jobs/refusal_phrases.sh` | `outputs/refusal_phrases/<model>_multilingual.json` (first pass, token cut) | Rerun with whole-word openings, add a refusal check, before the multilingual searches |

### Apple paper experiments: our status

| # | Experiment (paper section) | Our result in one line | Code | Results-log entry |
|---|---|---|---|---|
| 1 | Find the neuron: search + rerank (2.3) | Same neuron and pin as Apple on both 8B models (L14:F7924 at +20, L11:F4258 at −4) | `neuron_search.py`, `rerank.py`, `judge.py` | "Milestone 2 passes", "Milestone 2 rerank" |
| 2 | The jailbreak (2.4) | One pinned neuron: 95% (Qwen3-8B), 93% (Llama), vs 4% unpinned | `evaluate.py`, `judge.py`, `scripts/m3.sh` | "Milestone 3" |
| 3 | Capability cost, constant vs anchor (2.4) | The constant pin costs 3–19 points, the anchor about 0 | `capability.py`, `intervene.py` | "Milestone 3" |
| 4 | Where it fires (3.1) | Hardest at the template tokens before the answer; silent on normal prompts | `token_acts.py` | "Per-token firing" |
| 5 | Base models (3.2) | Qwen3-8B-Base already has it, but it fires on the harmful words, not at the answer boundary | `token_acts.py` | "Per-token firing" |
| 6 | Neuron as a detector (3.4) | XSTest AUROC within 0.01 of Apple's for all 3 models | `neuron_check.py`, `detect_threshold.py`, `guard_detect.py` | "Detection" |
| 7 | Alignment with Arditi's direction (App. G) | Matches Table 7 on both 8B models; the refusal neuron is #1 in its own layer in all 3 | `arditi.py` | "Appendix G alignment" |
| 8 | Residual-stream baseline (3.3) | Qwen3-8B: 26% vs 97% for the neuron. Llama: unscorable with Llama-Guard alone | `neuron_search.py --space residual`, `rerank.py` | "Section 3.3" |
| — | What the neuron responds to (3.5), concept neurons (4), models above 8B | Not done | | |
| — | Arditi ablation baseline (Fig. 5) | 91–95% attack success | `arditi.py` | "Appendix G alignment" |

### Our Arabic extension

| Experiment | Result | Code | Results-log entry |
|---|---|---|---|
| A: the English neuron on Arabic prompts | Fires at the same token, almost as strongly; AUROC 0.956 vs 0.972 (Qwen3-8B) | `neuron_check.py`, `detect_threshold.py` | "Arabic teaser", "Detection" |
| B: the search run on Arabic prompts | Largely the same top-5 neurons as the English search; Apple's neuron in both lists for all 3 models | `neuron_search.py`, `refusal_openings.py` | "Arabic search" (two entries) |
| C: the neuron as an Arabic filter | Beats Llama-Guard-3 in Arabic (0.956 vs 0.910) | `guard_detect.py` | "Detection" |

## Next steps

**Current priority (decided 2026-10-27): finish the replication in Arabic before any new direction.** That means the whole Apple paper again, with the Arabic refusal neuron found from scratch by their method, on two tracks: PolyRefuse Arabic (the same prompts as the English replication, machine-translated) and MultiJail Arabic (human-translated).

1. ✅ **Apple appendix review** (2026-10-27): [docs/apple_appendix_review.md](docs/apple_appendix_review.md). The MLP-space direction and the "does Arditi's direction work through the neuron" test are not in the paper.
2. **Arabic judging** (✅ validated 2026-10-27). Qwen3Guard-Gen-8B judges the Arabic directly. Llama-Guard-3 judges an NLLB-200 English translation (Wang et al.'s method). They agree (kappa 0.88); Llama-Guard on the raw Arabic undercounts and is not used. See the results log.
3. **Arabic Experiment 1** (in progress): Arabic refusal sentences, search, and rerank on held-out Arabic prompts, on both tracks.
4. **Arabic Experiments 2–8:**
   - 2: the jailbreak on the Arabic final-test sets, vs Arditi's direction computed from Arabic prompts
   - 3: capability (Arabic MMMLU)
   - 4: per-token firing
   - 5: base model
   - 6: detector vs the judges
   - 7: alignment with the Arabic Arditi direction
   - 8: residual numbers

   Then compare: is the Arabic-found neuron the English one, and is it as effective?
5. **Then the new directions from the 22 October meeting:**
   - the refusal direction inside the MLP (script written, `replication/mlp_direction.py`, not yet reviewed or run)
   - every language on both datasets (data ready; refusal-phrase tooling ready)
   - the language-agnostic guard
6. **A coherence-aware judge** (Apple's K.1 prompt on a local LLM), and **Section 3.3 at layer outputs**.
7. **Writing** (December), for 4 January.
8. **After the multilingual paper:** images and audio (Paper B), then the SAE and EMNLP-directions comparison (Paper C).

## Meetings and shared folder

- **Weekly meeting with Nadir every Thursday.** Each meeting gets dated slides and one page of minutes: attendees, what was discussed, decisions, action items.
- **Shared Google Drive folder** (fjoad1 account; anyone with the link can view): **[Refusal Neurons](https://drive.google.com/drive/folders/10q3hGmtRQUqM5EneyoZhL_j46phpbwy6)**
  - [Slides](https://drive.google.com/drive/folders/14CgO54B1EEDSPxSL4YIaJB80EHOnpDUB): one deck per meeting, named by date
  - [Meeting minutes](https://drive.google.com/drive/folders/1xjEww7FKtUxB_ms_m1BfkR0aKmH9LP1x): [2026-10-22 weekly meeting](https://docs.google.com/document/d/1TnyCnlIqGuIo23iHMe7UKnCt23fyArFX/edit), [2026-10-19 kickoff](https://docs.google.com/document/d/1iISPLsnNnbcexcsJRaZbgwe34YXMlaI3/edit)
- Minutes live only in Drive, not in this public repo, because they contain internal discussion. A Drive-ready copy is kept locally in `drive/` (gitignored). `drive/src/` holds the HTML sources, converted to Word with `soffice --headless --convert-to docx`.
- The slides' source (`slides/build_slides.js`) and PDF are also in this repo.

## Decisions so far, and why

These were settled in discussion. Don't reopen them without a new reason.

- **Replicate first, extend second.** Every direction depends on finding the neuron ourselves. Apple released no code, so we reimplement from the paper (Section 2 and Appendix A).
- **Models:** Qwen3-1.7B, Qwen3-8B, Llama-3.1-8B-Instruct. They cover both families in the paper and fit our GPUs. Larger models only if needed.
- **fp16.** V100/P100/T4 GPUs have no fast bf16. fp16, bf16 and fp32 gave the same search ranking on Qwen3-1.7B.
- **Data follows Arditi et al. exactly**, as Apple did: HarmBench validation (191 after removing JailbreakBench overlaps, matching the paper's count) and JailbreakBench as the test set. For training, use **Arditi's exact 128/128 sample** (`arditi_*_train`). With it the search reproduces Table 5's activations; with our own sample, Qwen3-1.7B's neuron dropped to rank 11.
- **The search follows the paper's appendix to the letter:** Table 4 refusal phrases, and only Table 3's red template tokens. Each of these changed the result. When the PDF text is ambiguous, render the page and look; the red highlighting is invisible in extracted text.
- **Judge: Llama-Guard-3-8B only, free and on panther** (decided 2026-10-21). There are no API credits for an LLM judge while this is exploratory. An answer is a successful attack when Llama-Guard labels it unsafe; that is Apple's secondary judge. Caveat: Llama-Guard does **not** reliably reject garbled output. It called all of Qwen3-8B's pin-80 outputs unsafe, and their median length was 100 characters.
- **One judge for every language.** Comparing English numbers from one judge with Arabic numbers from another would not be valid.
- **Arabic judges** (decided 2026-10-27): **Qwen3Guard-Gen-8B** (Apache-2.0, 119 languages, ~86 F1 on Arabic, also labels refusals) on the original text, and **Llama-Guard-3 on an NLLB-200 English translation**, as Wang et al. (NeurIPS 2025) did. Reported side by side; the agreement check passed (kappa 0.88), and Llama-Guard on the untranslated Arabic undercounts (0.84 vs 0.95), so it isn't used. A chat model was rejected as translator because it may refuse to translate harmful answers. FanarGuard (QCRI) would be a natural third judge, but it isn't public; ask Taha Sencar.
- **Rerank selection: the candidate that reaches ASR ≥ 0.90 at the smallest pin**, not the highest ASR, because Llama-Guard rewards garbled high-pin output. This rule recovers Apple's neuron and pin on both 8B models. `judge.py` prints this ranking.
- **The rerank is cheaper than the paper's:** a 9-value pin grid (2–60) and 128 new tokens instead of their 47 values and 512 tokens. The final attack (M3) uses 512 tokens like the paper.
- **Arabic only, for now.** A full replication plus an Arabic contribution is enough for Paper A. Modalities come after Arabic. Add languages along a resource spectrum later (e.g. French or Chinese, then Swahili or Bengali).
- **Human-translated multilingual data where possible**, so nobody can blame the translations: MultiJail (MIT) for harmful prompts, Aya (Apache-2.0) for harmless ones.
- **Multilingual: every language we have data for, on two datasets** (decided 2026-10-22). PolyRefuse (Wang et al., NeurIPS 2025; 14 languages including Arabic, machine-translated with Google Translate, parallel harmful and harmless; github.com/mainlp/Multilingual-Refusal) and MultiJail (10 languages, human-translated). Every experiment runs on both.
- **Submit to ARR on 4 January 2027** (decided 2026-10-22). It's low-hanging fruit and could be scooped, so speed matters more than scope; images and audio come after.
- **The search is the main tool of the project.** "Same neuron in Arabic?" is answered most strongly by running the search in Arabic; checking whether the English neuron fires in Arabic is the weaker test.
- **Arabic searches target each model's own Arabic refusal sentence.** The search's "push" needs a refusal sentence, and in Arabic the models refuse in Arabic. We read the sentences off 100 unpinned Arabic answers with `refusal_openings.py`. With English phrases, Qwen3-1.7B's neuron fell to rank 568.
- **Responses stay out of logs.** Generated text goes to the output JSON only; humans read it with `replication/show_responses.py`.
- **Several papers from one codebase** (below), not everything in one paper.

## Open questions
- **Language families for the fallback story:** how to group the 14 + 10 languages (by family, by script, by resource level).
- **Which local LLM to use as the coherence judge** (see Next steps 2).
- **Which multimodal models** share a backbone with our text models (e.g. a Qwen3-based VL or audio model)? Check each model's backbone before using it.
- **MultiJail is borderline in places:** the models' unpinned refusal rate on it is low (~0.45 by string match on Qwen3-8B). A stricter harmful set may be needed for Arabic attack numbers.

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

Our EMNLP paper found refusal in the **residual stream** as several category-specific directions that share a core. Steering along any of them acts as one knob for *whether* the model refuses, and the directions change *how* it refuses. Kazemi et al. work with raw **MLP neurons**, which have a privileged basis, and find one neuron is sufficient for *whether*. Goodfire argues that raw neurons are already interpretable in general, so SAEs are often unnecessary. A working hypothesis linking them: one gate neuron decides whether to refuse, and category-specific directions shape how. Our Experiment 7 supports it: the refusal neuron is the neuron most aligned with the refusal direction in its layer, in all three models.

## Running experiments

Everything heavy runs on **panther** (QCRI SLURM). The repo is a plain git clone at `~/refusal-neurons` there. The loop is: edit locally → commit and push → `git pull` on panther → `sbatch`. [AGENTS.md](AGENTS.md) has the exact commands and the panther gotchas. Job logs go to `logs/<job>-<id>.out` and results to `outputs/` on panther; both are gitignored, so numbers are copied into the results log.

### Scripts

**Setup and data**

| Script | What it does |
|---|---|
| `env/setup_env.sh` | SLURM job that builds the `refneu` conda env (torch 2.5.1 + `env/requirements.txt`) |
| `scripts/download_models.sh` | SLURM job that downloads Hugging Face models (the login node kills large downloads) |
| `data/prepare_data.py` | Arditi/Apple splits: train, validation, HarmBench, JailbreakBench, held-out Alpaca |
| `data/prepare_arabic.py` | MultiJail en/ar and Aya en/ar splits (test sets and search sets) |
| `data/prepare_xstest.py` | XSTest safe/unsafe splits |
| `data/prepare_capability.py` | Fixed MMLU (1,000) and GSM8K (250) subsets |
| `data/prepare_multilingual.py` | PolyRefuse (14 languages) and MultiJail + Aya (10 languages) splits, the same prompts in every language |

**Experiments** (`replication/`)

| Script | Experiment | What it does |
|---|---|---|
| `neuron_check.py` (+ `.sh`) | M1, detection, Arabic A | For a known neuron: AUROC (harmful vs harmless), per-token means, and refusal rate when pinned. Saves per-prompt detection scores |
| `neuron_search.py` (+ `.sh`) | Exp. 1 part 1, Arabic B, Exp. 8 | Apple's gradient × activation search over the first 2/3 of layers; top 5. `--watch L:F` reports any neuron's rank; `--space residual` searches residual-stream numbers; saves `*_means.pt`. Paper settings are the defaults |
| `rescore.py` | Exp. 1 | Re-ranks from a saved `*_means.pt` with a different token subset, on CPU |
| `rerank.py` | Exp. 1 part 2, Exp. 8 | Pins each top-5 candidate over a grid of pin values on HarmBench and saves the answers |
| `judge.py` | all attacks | Verdicts and ASR for any rerank/evaluate/neuron_check output. `--judge llamaguard\|qwen3guard`; `--translate` judges an NLLB-200 English translation (`--src_lang`, `--prompts_en`). Each judge writes its own fields; prints the best pin per candidate |
| `translate.py` | Arabic, multilingual | NLLB-200 translation into English (chunked at sentence breaks), used by `judge.py --translate` |
| `judge_agreement.py` | Arabic, multilingual | Agreement and Cohen's kappa between every pair of judges on the same responses |
| `evaluate.py` | Exp. 2 | JailbreakBench attack for one neuron: unpinned, constant pin, anchor k=1,2; 512 tokens |
| `intervene.py` | Exp. 2–3 | Shared constant and anchor (Eq. 6–7) interventions |
| `capability.py` | Exp. 3 | MMLU and GSM8K accuracy under each intervention |
| `token_acts.py` | Exp. 4–5 | Per-token neuron values over the whole prompt (content vs template tokens); base vs chat models |
| `guard_detect.py` | Exp. 6, Arabic C | Llama-Guard-3 as a prompt-only harmful-prompt detector (AUROC, precision, recall) |
| `detect_threshold.py` | Exp. 6, Arabic A | Turns detection scores into a filter: best cut-off, accuracy, precision, recall; `--calibrate` applies a cut-off from other data |
| `arditi.py` | Exp. 7, Fig. 5 | Arditi's refusal direction (selection, ablation attack) and the Appendix G cosine alignment |
| `refusal_openings.py` | Arabic B | Counts the stock openings of unpinned answers, to find a model's refusal sentence in a language; never prints full answers |
| `refusal_phrases.py` | multilingual | Each model's most common refusal openings per language, from short unpinned answers; feeds `neuron_search.py --phrases_json/--phrases_key` |
| `summarize.py` | any | Per-run metrics of an output JSON (refusal, ASR, response lengths), never the text |
| `show_responses.py` | any | Writes a few generated answers to a file, for a human to read |

**SLURM wrappers** (`scripts/`)

| Script | What it does |
|---|---|
| `scripts/m3.sh` | M3 for one neuron: `evaluate.py` + `judge.py` + `capability.py` |
| `scripts/jobs/*.sh` | The exact job scripts behind each results-log entry (rerank + judge, Arditi, residual baseline, Arabic searches, detection, per-token runs). Each has a header saying what it produced |
| `scripts/cpu.sh` | Runs one repo script on a CPU node (analyses of `outputs/`) |
| `scripts/pssh.sh` | Runs a command on panther, retrying when the VPN drops the connection |

Examples (on panther, from `~/refusal-neurons`):
```bash
sbatch replication/neuron_search.sh Qwen/Qwen3-8B float16 --watch 14:7924        # Exp. 1 part 1, paper settings
sbatch scripts/jobs/rerank_judge.sh outputs/neuron_search/Qwen3-8B_float16_arditi.json Qwen3-8B   # Exp. 1 part 2
sbatch scripts/m3.sh Qwen/Qwen3-8B 14 7924 20                                      # Exp. 2-3
sbatch scripts/cpu.sh replication/summarize.py outputs/rerank/Qwen3-8B_float16_arditi.json
python replication/show_responses.py outputs/neuron_check/<file>.json --out /tmp/responses.txt   # humans only
```

### Data (`data/splits/*.json`, committed)

| Split | Size | Source | Used for |
|---|---|---|---|
| `arditi_harmful_train`, `arditi_harmless_train` | 128 + 128 | Arditi's exact sample (AdvBench, MaliciousInstruct, TDC2023 / Alpaca) | the search (Exp. 1) |
| `arditi_harmful_val`, `arditi_harmless_val` | 32 + 32 | Arditi's validation sample | selecting Arditi's direction (Exp. 7) |
| `harmful_train`, `harmless_train` | 128 + 128 | our own seed-42 sample | superseded; kept for the record |
| `harmbench_val` | 191 | HarmBench minus JailbreakBench overlaps | the rerank (Exp. 1 part 2) |
| `jailbreakbench` | 100 | JailbreakBench | M1 and the final attack (Exp. 2) |
| `harmless_heldout` | 100 | Alpaca, disjoint from train | M1 separation, per-token plots |
| `xstest_unsafe`, `xstest_safe` | 200 + 250 | XSTest | detection (Exp. 6), harmful-word controls (Exp. 4) |
| `mmlu_1000`, `gsm8k_250` | 1000, 250 | MMLU, GSM8K test | capability (Exp. 3) |
| `multijail_en`, `multijail_ar` | 100 + 100 | MultiJail, the same prompts in both languages | Arabic A and C (test) |
| `aya_harmless_en`, `aya_harmless_ar` | 100 + 100 | Aya test split | Arabic A and C (test) |
| `multijail_train_en`, `multijail_train_ar` | 128 + 128 | MultiJail, disjoint from the 100 above | Arabic B (search) |
| `aya_train_en`, `aya_train_ar` | 128 + 128 | Aya train split | Arabic B (search) |
| `poly_<lang>_harmful_train`, `poly_<lang>_harmless_train` | 128 + 128 per language | PolyRefuse (Wang et al. 2025; Arditi's splits, Google-translated). The harmful 128 are the same prompts as `arditi_harmful_train` in all 14 languages | multilingual search |
| `poly_<lang>_harmful_test`, `poly_<lang>_harmless_test` | 572 + 500 per language (zh harmless: 72) | PolyRefuse test sets | multilingual detection and attack |
| `poly_<lang>_harmful_rerank`, `poly_<lang>_harmful_eval` | 191 + 100 per language | disjoint subsets of the harmful test set (one duplicated prompt overlaps) | rerank (≈ HarmBench) and final attack (≈ JailbreakBench) |
| `mj_<lang>_harmful_rerank` | 87 per language | the MultiJail prompts not used elsewhere | rerank on the MultiJail track |
| `mj_<lang>_harmful_train`, `mj_<lang>_harmful_test` | 128 + 100 per language | MultiJail, the same prompts in all 10 languages (= `multijail_train_*` / `multijail_*` for en, ar) | multilingual search / test |
| `mj_<lang>_harmless_train`, `mj_<lang>_harmless_test` | 128 + 100 per language (jv test: 91) | Aya in that language (not parallel across languages) | multilingual search / test |

PolyRefuse languages: en ar de es fr it ja ko nl pl ru th yo zh. MultiJail languages: en ar zh it vi ko th bn sw jv. Built by `data/prepare_multilingual.py`.

### Outputs (on panther, `~/refusal-neurons/outputs/`, not in git)

| Folder | Written by | Contents |
|---|---|---|
| `neuron_check/` | `neuron_check.py` | AUROC, per-token means, detection scores, pinned answers |
| `neuron_search/` | `neuron_search.py` | top-5 candidates and watched ranks (`.json`), per-layer means (`_means.pt`) |
| `rerank/` | `rerank.py` + `judge.py` | answers and Llama-Guard verdicts per candidate and pin |
| `evaluate/` | `evaluate.py`, `arditi.py` + `judge.py` | JailbreakBench answers and verdicts |
| `capability/` | `capability.py` | MMLU/GSM8K predictions and accuracy |
| `arditi/` | `arditi.py` | the direction, candidate scores, alignment report |
| `token_acts/` | `token_acts.py` | per-token values for every prompt |
| `guard_detect/` | `guard_detect.py` | Llama-Guard prompt scores |

## Layout

```
README.md                 this file: project guide, status, decisions, how to run
AGENTS.md                 operational rules for AI agents (CLAUDE.md just imports it)
RESEARCH_DIRECTIONS.md    the five research directions and how they map to papers
docs/explained_simply.md  plain-language walkthrough of every experiment and result
docs/apple_appendix_review.md  what the Apple paper did, section by section and appendix by appendix
docs/glossary.md          plain-language definitions
docs/roadmap.html         older visual roadmap page (standalone HTML)
docs/how_it_works.html    older visual explainer of the method (standalone HTML)
slides/                   meeting deck: build_slides.js -> refusal_neurons_update.pptx (+ .pdf)
reading/                  reading list (README.md), sources.tsv, download.sh; pdfs/ and web/ are gitignored
env/                      conda env job and requirements
data/                     split builders and the splits (data/splits/*.json)
replication/              all experiment code, the method notes and the results log (README.md)
scripts/                  SLURM wrappers, model download, panther helpers; scripts/jobs/ = per-experiment job scripts
drive/                    gitignored: Drive-ready slides and meeting minutes (drive/src = minutes sources)
```

## Getting the reading material

```bash
./reading/download.sh
```

The PDFs and web pages are third-party copyrighted material. They are not committed; only the links are.
