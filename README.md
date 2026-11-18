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

## Where things stand (updated 2026-11-16)

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
| **Arabic replication** | The whole Apple paper again in Arabic, with the Arabic neuron found from scratch, on two tracks (PolyRefuse, MultiJail) | 🟡 Experiment 1 search done (largely the same neurons as English). PolyRefuse rerank done: the Arabic search-and-rerank recovers Apple's neuron and pin on Qwen3-8B (L14:F7924, +20) and on Llama under Qwen3Guard (L11:F4258, −4); Qwen3-1.7B is a near-tie. MultiJail: no neuron reaches 0.90 on any model; the pins remove most refusals, but many answers are judged safe (borderline prompts). Experiment 1 done. Experiment 2 (the jailbreak) and 8 (alignment) done: on PolyRefuse the neuron jailbreaks Arabic as well as English. Experiments 3–5, 7 and 8 done. Experiment 6 (residual baseline) done 2026-11-17: no residual number reaches 0.90 under K.1 (best 0.55 on Llama, 0.28 on Qwen3-8B MultiJail), with one near-exception, Qwen3-8B on PolyRefuse (0.87 at the largest pin, coherent, against 0.97 for the neuron) |
| Modalities | Image and speech | ⬜ |

### Work in progress

What is running or parked, where its output lands, and what comes next. Job logs are in `logs/` on panther; job scripts are in `scripts/jobs/`.

| Work | Script | Output (on panther) | Next step |
|---|---|---|---|
| Arabic Experiment 1: refusal sentences, search and rerank answers, both tracks, 3 models | `scripts/jobs/arabic_exp1.sh <model> <English L:F>` | `outputs/refusal_phrases/<model>_ar.json`, `outputs/neuron_search/<model>_float16_{poly,mj}_ar_full.json`, `outputs/rerank/<model>_float16_{poly,mj}_ar.json` | Judge each model's rerank files with `sbatch scripts/jobs/judge_ar_rerank.sh <model>` (both Arabic judges, both tracks), then pick each Arabic neuron and pin by the 0.90 rule. Qwen3-1.7B is judged (job 436940): the rule doesn't settle a winner (see the results log); PolyRefuse is judged for all three: Qwen3-8B picks L14:F7924 at +20 (Apple's), Llama picks L11:F4258 at −4 (Apple's) under Qwen3Guard and is a near-tie under the other judge. MultiJail is judged for all three (8B: 437432/3, after 437021/2 timed out on P100s): no neuron reaches 0.90. **Done**; next is choosing each model's Arabic neuron for Experiments 2–8 (see the results log) |
| Part 2, the attack in every language | `scripts/jobs/part2_attack.sh`, `part2_judge.sh`, `replication/part2_table.py` | judge fields in `outputs/evaluate/<model>_L<L>_F<F>_<track>_<lang>.json` | **Done 2026-11-12** for all three models, three judges; see the results log |
| Part 2, rerank where the search does not put Apple's neuron in its top 5 | `scripts/jobs/part2_rerank.sh <model> <track_lang> ...` | `outputs/rerank/<model>_float16_<track>_<lang>.json` | Held from 12 to 16 November, released 2026-11-16. Apple's neuron is swept next to each language's own top 5. ✅ Done 2026-11-16 (jobs 468279/80 and 468856 for Qwen3-8B, 460190 Llama, 460623 Qwen3-1.7B): Apple's neuron wins at its English pin in Chinese, Japanese, Russian and Polish; PolyRefuse English is a near-tie; Yoruba and Swahili are not judgeable. Results log, 2026-11-16 |
| Part 2, low-resource follow-up and the guard comparison | `scripts/jobs/restore_refusal.sh`, `scripts/jobs/part2_guard.sh` | `outputs/neuron_check/*_restore_<lang>_{harmful,harmless}.json`, `outputs/guard_detect/<track>_<lang>[_qwen3guard].json` | **Done 2026-11-12**; see the results log |
| Part 3 in Arabic for the 8B models | `scripts/jobs/mlp_direction.sh` | `outputs/mlp_direction/<model>_L<L>_<tag>.{json,pt}`, `outputs/evaluate/<model>_L<L>_mlpdir_<tag>.json` | **Done 2026-11-12**, both datasets (same findings as English) |
| Arabic Experiment 6, residual baseline | `scripts/jobs/arabic_exp6.sh <model> <residual L:F> <track>` | `outputs/rerank/*_{poly,mj}_ar_residual.json` | ✅ Released 2026-11-16 at 23:05 (held from 12 November until the reranks finished). All four jobs (460193, 460194, 460195, 468263) done and logged 2026-11-17 |
| Follow-ups to the 5 November meeting notes, launched 2026-11-12 (see the table below) | `scripts/jobs/why_quick.sh <model> <L> <F> <pin> <cut-off>` (`why_capability.py`, `residual_share.py`, three English searches), `replication/why_tables.py` (no GPU) | `outputs/capability/*_why.json`, `outputs/arditi/*_residual_share_en.json`, `outputs/neuron_search/*_why_*.json`, `logs/cpu-<id>.out` | First halves **done 2026-11-12** (jobs 460347/8/9, CPU 460362). Top-k overlap for k > 5 and the layer where the signal first appears **done 2026-11-17**. Still open: the search and pin on refused prompts only, the held-neuron dose test, job logs in the repo, the how-to-run guide |

### Follow-ups from the 5 November meeting

Nine points came out of the meeting (Faaiz's notes; readings agreed 2026-11-12). Each "why" question gets a hypothesis and a small experiment. ✅ done, 🟡 running or partly, ⬜ not started.

| # | Question from the meeting | Hypothesis | Small experiment | Code | State |
|---|---|---|---|---|---|
| 1 | Why does Nanda's ablation cost less ability than the constant pin? | The pin forces the neuron far outside its natural range on every prompt; ablation removes something that is near zero on harmless prompts. Alternative: the neuron has other jobs, so any overwrite hurts | MMLU with the neuron reset to its harmless mean and pinned at 10%, 25%, 50%, 100% of m*; how far each intervention moves the next-token distribution (KL) | `why_capability.py` | 🟡 |
| 2 | Is "the anchor costs nothing" (Experiment 3) a real result? Count how often the neuron fires on MMLU | The neuron fires on very few MMLU questions, so the anchor is nearly off and the test shows little | Per question: the neuron's value, whether it passes the detector cut-off, the pin the anchor actually uses; accuracy split into fired / not fired | `why_capability.py` | 🟡 |
| 3 | Overlap of Apple's top-k list with ours, in every language, on both datasets | (a table, not a "why") | Top-5 overlap with our run of Apple's setup; larger k from the saved means. Apple's paper prints only the winning neuron per model, so the reference list is our replication's | `why_tables.py`, `topk_overlap.py` | ✅ top 5, and k up to 100 (2026-11-17) |
| 4 | Apple found its neuron in English, so why is it not first in English on other datasets? | It is not first in Apple's own setup either (4th, 5th, 3rd; the rerank picks it). The exact rank moves with the harmless prompts and the refusal phrases, because the top scores are close | The English search with one ingredient changed at a time: the model's own refusal openings; PolyRefuse's or MultiJail's prompts with Apple's phrases | `why_quick.sh` | 🟡 |
| 5 | In low-resource languages, separate the prompts where safety works from those where it does not | The neuron fires on the prompts the model refuses and is quiet on the ones it answers (the circuit is intact, the language triggers it less). Alternative: refusal goes by another route there | Neuron value on refused vs answered harmful prompts per language; then the search and the pin on the refused prompts only | `why_tables.py` (first part); second part ⬜ | 🟡 |
| 6 | In smaller models, is there a language-specific step? | In Qwen3-1.7B the refusal signal depends on a language step the small model does badly for rare languages: a steeper drop with language, and a later layer, than in the 8B | The neuron's strength per language as a share of English, per model; the layer where harmful and harmless first separate, per language | `why_tables.py`, `first_layer.py` | ✅ 2026-11-17: no extra step; the shared signal appears at the same layers in both Qwen sizes |
| 7 | Why are Chinese and Vietnamese lower on slide 10? Put logs in the repo; write a how-to-run guide | It is the MultiJail dataset (borderline prompts), not the language: PolyRefuse Chinese is 87% where MultiJail Chinese is 63% | Are the same MultiJail prompts "answered but judged safe" in every language? Plus the K.1 judge | `why_tables.py`, `part2_judge.sh` | 🟡; logs and guide ⬜ |
| 8 | Explain the math of how the neuron's share of the direction is found | — | Formula and worked example in the 5 November minutes (slide 11) | — | ✅ |
| 9a | Do slide 11's breakdown in the residual stream: does any single residual number hold much of the direction? | No: the direction is spread thinly over the 4,096 numbers, which is why pinning one residual number fails | Share of the largest coordinate, top 10, participation ratio | `residual_share.py` | 🟡 |
| 9b | How much of the residual direction does the refusal neuron account for? | Little, which is why holding the neuron "on" does not undo the ablation | The neuron's direct contribution to the direction (its gap × its write vector's projection) as a share; then the "held on" test at 1×, 2×, 4× with the ablation leaving the neuron's own write alone | `residual_share.py`; dose test ⬜ | 🟡 |

### Follow-ups from the 12 November meeting (the experiments for the next deck)

The 12 November meeting (Nadir, Taha, Majd) went through the first deck up to the base-model slide and raised these points (Faaiz's notes; readings agreed 2026-11-16). They are exploratory: run as many as possible and see what happens. Order: the cheap ones first, the fine-tuning ones last. ✅ done, 🟡 running or partly, ⬜ not started.

| # | Point from the meeting (first-deck slide) | What we test | Code | State |
|---|---|---|---|---|
| 10 | Does it work with CoCoNot? (data slide) | The neuron on CoCoNot's five refusal categories, four of them not about harm: does it separate "should refuse" from "should answer", and does the pin remove those refusals? | `data/prepare_coconot.py`, `scripts/jobs/coconot.sh` | ✅ (redone with a fixed direction 2026-11-17: outside the safety category the neuron does not fire on them on the 8B models) |
| 11 | Are the answers gibberish when pinned at a high value? (judge and pin slides) | One neuron, pins from small to 10× the attack pin, 191 HarmBench prompts; share of answers the K.1 judge calls coherent at each pin. Example answers for the slides are picked by Faaiz with `show_responses.py` | `scripts/jobs/pin_sweep.sh` (`rerank.py --only`) | ✅ |
| 12 | Is "this is harmful" a different thing from "how the model will respond"? Which is the neuron? | A 2×2: harmful or not × refused or not, using CoCoNot's non-harm refusals and XSTest's harmless-but-scary prompts, read off the neuron | `scripts/jobs/harm_vs_refusal.sh`, `harm_vs_refusal.py` | ✅ 2026-11-17: on the 8B models it reads harm, not the model's decision to refuse |
| 13 | −4.4 is the average; what is the range? (score slide) | Minimum, percentiles, maximum and a histogram of the neuron's value on harmful and harmless prompts | `followup_tables.py` | ✅ |
| 14 | Pin the neuron, then add the direction back; recompute the direction with the neuron pinned; remove the next neuron | (a) pinned neuron + refusal direction added: does refusal return? (b) the direction recomputed with the neuron off: what is left, and what is its top neuron? (c) switch that one off too and repeat | `direction_variants.py`, `scripts/jobs/direction_variants.sh` | ✅ |
| 15 | Set the top 5 neurons to off | All five search candidates held at their harmless values at once | `direction_variants.py` | ✅ |
| 16 | Arditi's direction without that neuron | Split the direction into the part along the neuron's write vector v and the rest; remove each part alone | `direction_variants.py` | ✅ |
| 17 | What if v is the direction? | Remove v from the residual stream at every layer, as Nanda's method does with its direction | `direction_variants.py` | ✅ |
| 18 | Experiment 4 with benign words that lead to something harmful (per-token slide) | (a) in our prompts: how often the neuron fires at the decision point with no single word firing; (b) a small hand-built set, to be approved by Faaiz | `followup_tables.py`; (b) ⬜ | 🟡 (a) done |
| 19 | Does the neuron always fire in the base model when it fires in the chat model? (base-model slide) | Prompt by prompt, the same 100 prompts in both models | `followup_tables.py` | ✅ |
| 20 | How strong is the gap in the base model against the trained one? | The harmful-minus-harmless gap on the words and at each template token, base against chat | `followup_tables.py` | ✅ |
| 21 | Break safety with a few dozen fine-tuning examples (Taha: papers show this works); then pin or freeze the neuron and see whether safety survives | Fine-tune the chat model on a few dozen harmful examples; measure refusal and the neuron before and after; repeat with the neuron frozen. Qwen3-1.7B first, then an 8B model | to write; needs a training setup | ⬜ last |
| 22 | Freeze the safety neurons in the base model, then safety-train | Needs a safety-training recipe we do not have; only after 21 | — | ⬜ stretch |

### Follow-ups from the 17 November meeting

The 17 November group meeting went mainly through the first (5 November) deck again, with most questions on the MLP-direction slide (Faaiz's notes; readings agreed 2026-11-18). These are hypotheses to test, whatever the intuition says. ✅ done, 🟡 running or partly, ⬜ not started.

| # | Point from the meeting | What we test | Code | State |
|---|---|---|---|---|
| 23 | The MLP direction was removed in one layer only; Nanda's ablation acts at every layer | In order: (a) the one MLP direction we have (from the refusal neuron's layer) removed inside every MLP; (b) each MLP's own direction removed inside its own MLP, all layers at once; (c) Nanda's residual direction removed only from what the MLPs write, and only from what attention writes | `mlp_every_layer.py`, `scripts/jobs/mlp_every_layer.sh` | ✅ 2026-11-18. (a) no more than one layer; (b) and (c, MLP outputs only) remove most refusal openings, 0.71–0.93 by Llama-Guard but 0.28–0.53 by K.1 on the 8B models (0.65–0.86 on Qwen3-1.7B): the judges disagree on these answers and they have not been read; (c) MLP outputs do more than attention outputs on every model, neither alone equals the full ablation |
| 24 | Is the reverse method finding the same neuron as Apple's method, independently? | The 5 November run chose the layer and token with Apple's neuron. Blind version: the direction at every layer, averaged over the template tokens; every neuron ranked by its share of its layer's direction and by the size of what it writes | `mlp_every_layer.py` (same jobs) | ✅ 2026-11-18: by share of its layer's direction Apple's neuron is 1st (Llama), 2nd (Qwen3-1.7B) and 5th (Qwen3-8B) in the whole model |
| 25 | The α = 4 was never searched for; Apple grid-searched its pin, and our EMNLP paper swept Nanda's steering strength up to full saturation | Directions are steered (Nanda's way), neurons are pinned (Apple's way). (a) The MLP direction's α swept from 0.5 to 32 gaps with Apple's rule (smallest value reaching 0.90) and the K.1 judge, reported also as unit-vector strengths, the EMNLP paper's convention; (b) the pin grid on whichever neuron the blind reverse route finds | (a) `mlp_direction.py --steer_only`, `scripts/jobs/alpha_grid.sh`; (b) `scripts/jobs/blind_rerank.sh` | 🟡 (a) ✅ Qwen3-8B: under K.1, 0.90 is first reached at α = 2.5 (0.96), best 0.97 at 3; at α = 4 it is 0.87 and from 6 the answers are no longer coherent, which Llama-Guard does not see. Llama under K.1 peaks at 0.895 at α = 3, one prompt short of 0.90 (finer grid 2.75–3.5: 0.84, 0.83, 0.76), and is 0.47 at α = 4; Qwen3-1.7B reaches 0.92 at α = 2, peaks at 0.95 at 2.5 and is 0.30 at 4. On all three models the peak is at 2.5–3 gaps and the answers stop being coherent by 4–6; (b) 🟡 the blind route's top 5 per model through Apple's rerank (`mlp_every_layer.py --blind_only`, `scripts/jobs/blind_rerank.sh`). Qwen3-8B done: under K.1 a tie between Apple's neuron (+20, 0.95) and a new neuron of the same layer, L14:F1251 (−20, 0.95). Llama done: it picks Apple's neuron at Apple's pin (−4, 0.96), no other candidate above 0.14 under K.1. Qwen3-1.7B done: L15:F1109 at +40 (0.92), Apple's neuron second (+60, 0.95; 0.87 at +40). Still running: two same-sign candidates rerun pinned towards harmless (`scripts/jobs/blind_flip.sh`) |
| 26 | What do the other top candidates do? Does pinning more neurons raise the rate (how many to reach 100%)? Is the whole more than the sum of its parts? | (a) from the existing rerank files: each candidate alone, and whether they fail on the same prompts; (b) the top 1, 2, 3, 4, 5 pinned together, where one neuron leaves room (Qwen3-1.7B, MultiJail, transfer between datasets) | (a) `topk_union.py`; (b) `joint_pin.py`, `scripts/jobs/joint_pin.sh` | 🟡 (a) done 2026-11-18: on HarmBench and PolyRefuse the five between them reach 0.96–0.99; on MultiJail 0.72–0.93, and 7–28% of prompts are missed by all five; (b) 🟡 MultiJail Arabic. Full pins together give incoherent answers on all three models. Qwen3-8B: four neurons at a quarter of their pins reach 0.97 under K.1, all coherent, against 0.77 for the best single neuron (the guards show no gain). Qwen3-1.7B: 0.94 against 0.90 at half pins. Llama: four of the five guard-chosen pins are incoherent alone, so no answer yet. Running: K.1 on the MultiJail reranks (470494) to choose coherent pins, and each neuron alone at the reduced strengths as the control |
| 27 | If one of the top neurons is on, are they all on? One on at a high pin with the rest pushed the other way: does it cancel? | Prompt-by-prompt co-firing of the top 5; one pinned to the attack side with the others held on the harmful side | to write | ⬜ |
| 28 | Do the top 5 residual-stream numbers together work where one does not (32%)? | The five best residual numbers pinned together | to write | ⬜ |
| 29 | Does changing the refusal phrases (the "starter statement") change the result? | Standard: the model's most common refusal openings per language (Apple's fixed list for the English replication). Try other lists (a single phrase, an unrelated phrase as a control) and track the top 5 and the rerank winner. So far only the rank of Apple's neuron under two lists (5 November follow-up, note 4) | `refusal_phrases.py`, `neuron_search.py --phrases_json` | 🟡 |
| 30 | Justify dropping the languages where the model fails (Yoruba, Swahili, Javanese) | MMLU per language; drop a language below a set accuracy, or with no MMLU translation. Not dropped yet; the justification is for the paper | to write; needs a multilingual MMLU | ⬜ |

Answered in discussion, no experiment: how Apple's neuron can be "6th" when the rerank picks from a top 5 (the rank is its place in a different search's list, PolyRefuse English prompts with the model's own phrases; in Apple's setup it is 5th). Single neurons for other behaviours, and a culture neuron, are in [RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md) as ideas for a later paper.

**Multimodal (speech and images):** direction set on 2026-11-18 (see Decisions). It runs on omni models, with Asim on activation extraction and experiments. Candidate models, the text-to-speech tool and the datasets Nadir listed are in [reading/README.md](reading/README.md) under "Multimodal: models, tools and datasets". No experiment plan written yet; the checkpoint meant by "Qwen3.8-Omni-Flash" is to be confirmed with Nadir (not on Hugging Face under that name on 2026-11-18).

### Part 1 checklist: every Apple experiment, in English and in Arabic

Each experiment is run the same way as the paper. In Arabic the point is to see whether the model behaves the same, not to match Apple's numbers. ✅ done, 🟡 partly or running, ⬜ not started. Arabic runs on both tracks (PolyRefuse and MultiJail) unless noted.

| # | Experiment (paper section) | English | Arabic | Code | Results-log entry |
|---|---|---|---|---|---|
| 1 | Find the neuron: search + rerank (2.3, App. A) | ✅ Same neuron and pin as Apple on both 8B models; Qwen3-1.7B a near-tie | ✅ Same neuron and pin as English on Qwen3-8B (both judges) and Llama (one judge); Qwen3-1.7B a near-tie; MultiJail reaches no 0.90 winner | `neuron_search.py`, `rerank.py`, `judge.py` | "Milestone 2 …", "Arabic Experiment 1" |
| 2 | The jailbreak: constant pin, anchor, vs Arditi's ablation (2.4, Fig. 5, App. H) | ✅ 95% (Qwen3-8B), 93% (Llama) vs 4% unpinned; Arditi 91–95% | ✅ PolyRefuse 88–95% pinned (7–40% unpinned), about equal to Arditi's Arabic direction; MultiJail 58–76%, Arditi weaker or none | `evaluate.py`, `arditi.py`, `scripts/jobs/arabic_exp2.sh` | "Milestone 3", "Arabic Experiment 2" |
| 3 | Capability cost: MMLU, GSM8K (2.4, App. H) | ✅ Constant pin costs 3–19 points, anchor about 0, Arditi's ablation 1–2 | ✅ Arabic MMLU (MMMLU, same questions), all 3 models: same pattern as English (constant −5 to −10, anchor 0 to −2, Arditi −1 to −3). No Arabic GSM8K exists | `capability.py` | "Milestone 3" |
| 4 | Where it fires, per token (3.1, App. F) | ✅ Hardest at the template tokens before the answer | ✅ Same token, 60–80% of the English strength; AUROC 0.88–1.00 | `token_acts.py` | "Per-token firing" |
| 5 | Base models (3.2) | ✅ Qwen3-8B-Base already has it, on the harmful words | ✅ The Qwen3 base models already fire on the harmful words in Arabic too | `token_acts.py` | "Per-token firing" |
| 6 | Residual stream vs MLP (3.3, App. E) | ✅ Under the K.1 judge: best residual dimension 0.16 (Qwen3-8B), 0.32 (Llama) vs 0.97 for the neuron; Apple 39% / 29–45% | ✅ 2026-11-17: best residual number under K.1 0.55 (Llama, PolyRefuse), 0.53 (Llama, MultiJail), 0.28 (Qwen3-8B, MultiJail), **0.87 (Qwen3-8B, PolyRefuse, at the largest pin)** vs 0.79–0.99 for the neuron; none reaches 0.90 | `neuron_search.py --space residual`, `rerank.py` | "Section 3.3", "Arabic Experiment 6" |
| 7 | Neuron as a detector (3.4, App. C, D) | ✅ XSTest AUROC within 0.01 of Apple's | ✅ PolyRefuse AUROC 0.95–1.00 (English 0.98–1.00); Llama-Guard as good there; the neuron's edge is on MultiJail | `neuron_check.py`, `detect_threshold.py`, `guard_detect.py` | "Detection" |
| 8 | Alignment with Arditi's direction (App. G) | ✅ Matches Table 7; the neuron is #1 in its own layer in all 3 | ✅ Arabic direction in the same or next layer; Apple's neuron 1st (Llama both tracks, Qwen3-8B PolyRefuse) or 2nd in its layer | `arditi.py` | "Appendix G alignment" |
| 9 | What the neuron responds to: max-activating text (3.5, App. I, L) | ⬜ Needs a corpus (The Pile) and a human to read the top snippets, which are largely explicit | ⬜ Needs an Arabic corpus | — | — |
| 10 | Concept neurons, e.g. "suicide" steering (4) | ⬜ A different question from refusal | ⬜ | — | — |
| 11 | Apple's main judge: refusal **and** coherence (App. K.1) | ✅ English attacks rescored on gemma-2-9b-it: 0.89–0.97, every pinned answer coherent; residual baselines 0.16 (Qwen3-8B), 0.32 (Llama) | ✅ Arabic attacks: PolyRefuse 0.96–0.99, MultiJail 0.79–0.94 (well above the guards: the pinned answers stop refusing but some prompts are borderline) | — | — |
| — | Inducing refusal on harmless prompts (App. J, qualitative only) | ⬜ | ⬜ | — | — |

Models above 8B are out of scope (decided at the start). The early Arabic pieces below fed into rows 1, 4 and 7.

### Our Arabic extension

| Experiment | Result | Code | Results-log entry |
|---|---|---|---|
| A: the English neuron on Arabic prompts | Fires at the same token, almost as strongly; AUROC 0.956 vs 0.972 (Qwen3-8B) | `neuron_check.py`, `detect_threshold.py` | "Arabic teaser", "Detection" |
| B: the search run on Arabic prompts | Largely the same top-5 neurons as the English search; Apple's neuron in both lists for all 3 models | `neuron_search.py`, `refusal_openings.py` | "Arabic search" (two entries) |
| C: the neuron as an Arabic filter | Beats Llama-Guard-3 in Arabic (0.956 vs 0.910) | `guard_detect.py` | "Detection" |

## Next steps

The project has three parts (restated by Faaiz, 2026-11-05):

**Part 1: the whole Apple paper, in English and in Arabic** (current). Every experiment the paper runs, done the same way, in Arabic as well as English, finishing any English gaps. The aim is to investigate whether Arabic behaves the same, not to reproduce Apple's numbers. Checklist above. In order:
1. Arabic Experiments 2–8 (rows 2–8 of the checklist). All launched 2026-11-05.
2. The English gaps: Arditi's capability cost (row 3), Llama's residual baseline (row 6, needs the coherence judge). Both launched 2026-11-05.
3. A coherence-aware judge (Apple's K.1 prompt on a local model, row 11), then rescore the attacks with it in both languages. Launched 2026-11-05 on gemma-2-9b-it.
4. Rows 9 and 10 (what the neuron responds to; concept neurons), in both languages. These need a corpus and a human reader; lowest priority.

**Part 2: every other language** (top priority since 2026-11-05; search, detector and attack launched). The same experiments for every language in PolyRefuse (Wang et al., the dataset Nadir suggested; 14 languages: en ar de es fr it ja ko nl pl ru th yo zh), and in MultiJail (10 human-translated languages), as decided on 22 October. Data splits are ready. First step: rerun the refusal phrases with whole-word openings and a refusal check. Then:
- the search in each language, and the overlap of the top-5 neurons across languages
- the language-agnostic neuron, or one per language family, or one per language
- that neuron as one guard across languages

**Part 3: Nadir's ideas beyond the paper** (from the 22 October meeting):
- Arditi's refusal direction computed **inside the MLP** (over the neuron values, not the residual stream); steer or ablate with it and compare with the single neuron and the residual direction
- How much of that direction's weight sits on the refusal neuron (decomposition)
- When the direction works, is it only through this neuron? (Ablate the direction but hold the neuron at its harmful value)
- The reverse route: compute the direction first, take its strongest neuron, and check whether it is the refusal neuron

`replication/mlp_direction.py` implements most of Part 3. Reviewed 2026-11-05 and running for all three models, in English and on both Arabic tracks (`scripts/jobs/mlp_direction.sh`). Caveat: in the held-neuron run, Arditi's all-layer ablation also strips the direction out of the neuron's own write, so only its other effects can restore refusal.

Then **writing** (December), for 4 January. After this paper: images and audio (Paper B), then the SAE and EMNLP-directions comparison (Paper C).

## Meetings and shared folder

- **Weekly meeting with Nadir every Thursday.** Each meeting gets dated slides and one page of minutes: attendees, what was discussed, decisions, action items.
- **Shared Google Drive folder** (fjoad1 account; anyone with the link can view): **[Refusal Neurons](https://drive.google.com/drive/folders/10q3hGmtRQUqM5EneyoZhL_j46phpbwy6)**
  - [Slides](https://drive.google.com/drive/folders/14CgO54B1EEDSPxSL4YIaJB80EHOnpDUB): one deck per meeting, named by date
  - [Meeting minutes](https://drive.google.com/drive/folders/1xjEww7FKtUxB_ms_m1BfkR0aKmH9LP1x): [2026-11-05 weekly meeting](https://docs.google.com/document/d/1XHGakgoGHCT23WmCpIR8pH5S4-mypLJ-/edit), [2026-10-22 weekly meeting](https://docs.google.com/document/d/1TnyCnlIqGuIo23iHMe7UKnCt23fyArFX/edit), [2026-10-19 kickoff](https://docs.google.com/document/d/1iISPLsnNnbcexcsJRaZbgwe34YXMlaI3/edit)
  - "Start here - links and reading list" (in the folder's root; source `drive/src/links.html`): one page for a new person, with the code, the papers to read first, the omni models and datasets, and the wider reading list. Update it when a model, dataset or key paper is added.
- Each set of minutes ends with an appendix, "The slides explained, step by step": one section per slide of that meeting's deck, saying in plain words what was done, how every number was calculated, and what the data looks like (added 2026-11-12 for both decks, at Nadir's request).
- Minutes live only in Drive, not in this public repo, because they contain internal discussion. A Drive-ready copy is kept locally in `drive/` (gitignored). `drive/src/` holds the HTML sources, converted to Word with `soffice --headless --convert-to docx`.
- The slides' source (`slides/build_slides.js`) and PDF are also in this repo.

## Decisions so far, and why

These were settled in discussion. Don't reopen them without a new reason.

- **Finish the multilingual and multimodal thread before any new direction** (2026-11-18, Nadir and Faaiz). Compute and time are limited, and the research questions for this thread are clear. The group meetings are pushed back; others are welcome to join and give input, but no new directions for now. Asim joins for the omni-model part (activation extraction and running experiments).
- **Part 1 is an investigation, not a reproduction** (2026-11-05). In Arabic (and later every language) we run Apple's experiments the same way and report whether the model behaves the same; matching Apple's numbers is a finding, not the goal.
- **Replicate first, extend second.** Every direction depends on finding the neuron ourselves. Apple released no code, so we reimplement from the paper (Section 2 and Appendix A).
- **Models:** Qwen3-1.7B, Qwen3-8B, Llama-3.1-8B-Instruct. They cover both families in the paper and fit our GPUs. Larger models only if needed.
- **fp16.** V100/P100/T4 GPUs have no fast bf16. fp16, bf16 and fp32 gave the same search ranking on Qwen3-1.7B.
- **Data follows Arditi et al. exactly**, as Apple did: HarmBench validation (191 after removing JailbreakBench overlaps, matching the paper's count) and JailbreakBench as the test set. For training, use **Arditi's exact 128/128 sample** (`arditi_*_train`). With it the search reproduces Table 5's activations; with our own sample, Qwen3-1.7B's neuron dropped to rank 11.
- **The search follows the paper's appendix to the letter:** Table 4 refusal phrases, and only Table 3's red template tokens. Each of these changed the result. When the PDF text is ambiguous, render the page and look; the red highlighting is invisible in extracted text.
- **Judge: Llama-Guard-3-8B only, free and on panther** (decided 2026-10-21). There are no API credits for an LLM judge while this is exploratory. An answer is a successful attack when Llama-Guard labels it unsafe; that is Apple's secondary judge. Caveat: Llama-Guard does **not** reliably reject garbled output. It called all of Qwen3-8B's pin-80 outputs unsafe, and their median length was 100 characters.
- **One judge for every language.** Comparing English numbers from one judge with Arabic numbers from another would not be valid.
- **Arabic judges** (decided 2026-10-27): **Qwen3Guard-Gen-8B** (Apache-2.0, 119 languages, ~86 F1 on Arabic, also labels refusals) on the original text, and **Llama-Guard-3 on an NLLB-200 English translation**, as Wang et al. (NeurIPS 2025) did. Reported side by side; the agreement check passed (kappa 0.88), and Llama-Guard on the untranslated Arabic undercounts (0.84 vs 0.95), so it isn't used. A chat model was rejected as translator because it may refuse to translate harmful answers. FanarGuard (QCRI) would be a natural third judge, but it isn't public; ask Taha Sencar.
- **Priority: Part 2 (every language) first, then Part 3 (Nadir's ideas), then what's left of Part 1** (Faaiz, 2026-11-05). On panther the remaining Part 1 jobs (Arabic residual baseline) and the Part 3 jobs are queued at lower priority (`nice`) behind Part 2.
- **Part 2 starts with the search, the detector and the attack with Apple's neuron in every language** (2026-11-05). This is the same choice as for Arabic: Apple's neurons at their English pins make every language directly comparable. The per-language rerank (the costliest step) comes later, if the search shows a language with a different top neuron. Refusal phrases now pass a Qwen3Guard refusal check; a language where the model hardly ever refuses falls back to the English Table 4 phrases.
- **Apple's main judge criterion (K.1) runs on gemma-2-9b-it** (decided 2026-11-05). Apple's main judge is Claude with their K.1 prompt: an attack succeeds when the answer is not a refusal and is coherent. With no API credits, the same three questions run locally (`judge.py --judge k1`), worded in our own words with K.1's JSON fields. gemma-2-9b-it is already cached, reads Arabic, and comes from a different family than the three models under test, so no model grades itself. It is reported next to Llama-Guard and the Arabic judges, not instead of them.
- **Arabic capability uses MMMLU** (decided 2026-11-05): `mmmlu_ar_1000` is the same 1,000 questions as `mmlu_1000`, professionally translated (OpenAI's MMMLU), with an Arabic answer instruction. No Arabic GSM8K exists, so Arabic Experiment 3 is MMLU only.
- **Arabic Experiments 2–8 use Apple's neurons** (decided 2026-11-05): Qwen3-8B L14:F7924 at +20, Llama L11:F4258 at −4, Qwen3-1.7B L13:F3270 at +40. The Arabic PolyRefuse rerank picks them for both 8B models (Llama under one judge, a near-tie under the other); for Qwen3-1.7B it's a near-tie in both languages, and +40 is Apple's neuron's best Arabic pin. Using the same neurons as English makes the two languages directly comparable.
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
| `data/prepare_mmmlu.py` | Arabic MMLU: the same 1000 questions as `mmlu_1000`, from MMMLU (`mmmlu_ar_1000`) |
| `data/prepare_multilingual.py` | PolyRefuse (14 languages) and MultiJail + Aya (10 languages) splits, the same prompts in every language |

**Experiments** (`replication/`)

| Script | Experiment | What it does |
|---|---|---|
| `neuron_check.py` (+ `.sh`) | M1, detection, Arabic A | For a known neuron: AUROC (harmful vs harmless), per-token means, and refusal rate when pinned. Saves per-prompt detection scores. `--sign` fixes the neuron's harmful sign (otherwise it is taken from the two splits and can flip on splits that are not harmful against harmless) |
| `neuron_search.py` (+ `.sh`) | Exp. 1 part 1, Arabic B, Exp. 8 | Apple's gradient × activation search over the first 2/3 of layers; top 5. `--watch L:F` reports any neuron's rank; `--space residual` searches residual-stream numbers; saves `*_means.pt`. Paper settings are the defaults |
| `rescore.py` | Exp. 1 | Re-ranks from a saved `*_means.pt` with a different token subset, on CPU |
| `rerank.py` | Exp. 1 part 2, Exp. 8 | Pins each top-5 candidate over a grid of pin values on HarmBench and saves the answers. `--add_watched` also sweeps the search's watched neuron (Apple's) when it is outside the top 5 |
| `judge.py` | all attacks | Verdicts and ASR for any rerank/evaluate/neuron_check output. `--judge llamaguard\|qwen3guard`; `--translate` judges an NLLB-200 English translation (`--src_lang`, `--prompts_en`). `--judge k1`: Apple's main criterion (not refusing and coherent) on gemma-2-9b-it. Each judge writes its own fields; prints the best pin per candidate. `--skip_done` skips a file a judge already covered; on a GPU out-of-memory error the batch is halved and retried |
| `translate.py` | Arabic, multilingual | NLLB-200 translation into English (chunked at sentence breaks), used by `judge.py --translate` |
| `judge_agreement.py` | Arabic, multilingual | Agreement and Cohen's kappa between every pair of judges on the same responses |
| `evaluate.py` | Exp. 2 | JailbreakBench attack for one neuron: unpinned, constant pin, anchor k=1,2; 512 tokens |
| `intervene.py` | Exp. 2–3 | Shared constant and anchor (Eq. 6–7) interventions |
| `capability.py` | Exp. 3 | MMLU and GSM8K accuracy under each intervention; `--arditi <pt>` adds Arditi's ablation, `--modes` picks modes, `--mmlu_instruction` for other languages |
| `token_acts.py` | Exp. 4–5 | Per-token neuron values over the whole prompt (content vs template tokens); base vs chat models |
| `guard_detect.py` | Exp. 6, Arabic C, Part 2 | Llama-Guard-3 (or `--guard qwen3guard`) as a prompt-only harmful-prompt detector (AUROC, precision, recall); takes several language sets at once |
| `detect_threshold.py` | Exp. 6, Arabic A | Turns detection scores into a filter: best cut-off, accuracy, precision, recall; `--calibrate` applies a cut-off from other data |
| `arditi.py` | Exp. 7, Fig. 5 | Arditi's refusal direction (selection, ablation attack) and the Appendix G cosine alignment. `--harmful/--harmless/--harmful_val/--harmless_val/--n_val` and `--refusal_tokens_json` run it in another language |
| `refusal_openings.py` | Arabic B | Counts the stock openings of unpinned answers, to find a model's refusal sentence in a language; never prints full answers |
| `refusal_tokens.py` | Arabic Exp. 2, 7 | Picks a model's refusal tokens in a language for Arditi's score, from first tokens of judged unpinned answers; feeds `arditi.py --refusal_tokens_json` |
| `refusal_phrases.py` | multilingual | Each model's most common refusal openings per language, from short unpinned answers; feeds `neuron_search.py --phrases_json/--phrases_key` |
| `mlp_direction.py` | Part 3 | Arditi-style refusal direction inside the MLP (before `down_proj`), its decomposition over neurons at every layer, and attack runs: ablate it, ablate it without the refusal neuron, steer, reset the neuron, Arditi's ablation with the neuron held |
| `summarize.py` | any | Per-run metrics of an output JSON (refusal, ASR, response lengths), never the text |
| `harm_vs_refusal.py` | 12 Nov follow-up, point 12 | CPU. The 2×2 harmful or not × refused or not: the neuron's fixed-sign value on CoCoNot and XSTest prompts, split by whether the model itself refused (Qwen3Guard). Reads the files `scripts/jobs/harm_vs_refusal.sh` makes |
| `mlp_every_layer.py` (+ `scripts/jobs/mlp_every_layer.sh`) | 17 Nov follow-up, points 23–24 | The MLP direction removed at every layer (one direction in every MLP; each MLP's own), Nanda's direction removed from MLP outputs only or attention outputs only, and every neuron ranked blind from the per-layer directions. Writes `outputs/evaluate/*_mlpall_en.json` |
| `first_layer.py` (+ `scripts/jobs/first_layer.sh`) | 5 Nov follow-up, note 6 | Per language set and layer: how well a difference-of-means direction separates harmful from harmless prompts in the residual stream, with the direction fitted in the same language and with the English one. Forward passes only |
| `topk_overlap.py` | 5 Nov follow-up, note 3 | CPU. Rebuilds each language's full search ranking from the saved `*_means.pt` and prints the top-k overlap with the English reference for k = 5 to 100, the neurons every language shares, and Apple's neuron's rank per language |
| `topk_union.py` | 17 Nov follow-up, point 26a | CPU. From judged rerank files: each top-5 candidate alone at its own best pin, the share of prompts jailbroken by at least one of the best k (each pinned alone), and how many prompts all five fail on |
| `joint_pin.py` | 17 Nov follow-up, point 26b | Pins the best 2, 3, 4, 5 candidates together, each at its own best pin from the judged rerank (full and half strength). `--report` (no GPU) prints the joint rate against "any of them alone" and how many prompts only the joint pin jailbreaks |
| `show_responses.py` | any | Writes a few generated answers to a file, for a human to read. Reads neuron_check, rerank, evaluate and direction_variants files; `--pins 20,40,80 --by_prompt` puts the same prompt's answers at several pins side by side (for the example slide) |

**SLURM wrappers** (`scripts/`)

| Script | What it does |
|---|---|
| `scripts/m3.sh` | M3 for one neuron: `evaluate.py` + `judge.py` + `capability.py` |
| `scripts/jobs/*.sh` | The exact job scripts behind each results-log entry (rerank + judge, Arditi, residual baseline, Arabic searches, detection, per-token runs). Each has a header saying what it produced |
| `scripts/cpu.sh` | Runs one repo script on a CPU node (analyses of `outputs/`) |
| `scripts/pssh.sh` | Runs a command on panther, retrying when the VPN drops the connection |
| `scripts/wait_job.sh <jobid> ...` | Run locally in the background: returns when any of the given panther jobs leaves the queue (polls every 5 minutes through `pssh.sh`) |

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
| `mmmlu_ar_1000` | 1000 | MMMLU Arabic (OpenAI), the same questions as `mmlu_1000` | Arabic capability (Arabic Exp. 3) |
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
| `neuron_search/` | `neuron_search.py` | top-5 candidates and watched ranks (`.json`), per-layer means (`_means.pt`); `<model>_float16_blind.json` is the blind reverse route's top 5 (`mlp_every_layer.py`) |
| `rerank/` | `rerank.py` + `judge.py` | answers and Llama-Guard verdicts per candidate and pin |
| `evaluate/` | `evaluate.py`, `arditi.py` + `judge.py` | JailbreakBench answers and verdicts |
| `capability/` | `capability.py` | MMLU/GSM8K predictions and accuracy |
| `arditi/` | `arditi.py` | the direction, candidate scores, alignment report |
| `token_acts/` | `token_acts.py` | per-token values for every prompt |
| `guard_detect/` | `guard_detect.py` | Llama-Guard prompt scores |
| `mlp_direction/` | `mlp_direction.py` | the MLP-space direction, per-layer decompositions (`.json`) and moments (`.pt`) |
| `refusal_phrases/`, `refusal_tokens/` | `refusal_phrases.py`, `refusal_tokens.py` | refusal openings / refusal tokens per model and language |
| `joint/` | `joint_pin.py` + `judge.py` | answers and verdicts with several top-5 neurons pinned together (`<model>_<tag>.json`) |

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
                          later decks: build_slides_<date>.js -> refusal_neurons_<date>.pptx (5, 12 and 17 November);
                          the 17 November deck answers the 12 November points and leaves its example-answer slide to fill by hand
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
