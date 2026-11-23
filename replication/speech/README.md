# Speech track: the refusal neuron for typed and spoken requests (Qwen2.5-Omni-7B)

Asim's part of the multilingual-multimodal thread. Question: is refusal gated by the **same single neuron**
when a harmful request is **typed** or **spoken**? Scope (2026-11-23): text and speech only.

Results: [replication/README.md](../README.md), entries **2026-11-23: Speech rerun with the stop token (v2)**
(current numbers) and **2026-11-19: Speech, first results** (first run, superseded).

## Design

- **Model:** Qwen/Qwen2.5-Omni-7B, the thinker only (`Qwen2_5OmniThinkerForConditionalGeneration`), bf16, speech
  output off. 28 decoder layers × 18,944 MLP neurons; the search covers the first 18 layers (340,992 neurons).
- **System prompt:** Qwen2.5-Omni's documented default ("You are Qwen, a virtual human developed by the Qwen Team,
  Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."), the
  same for typed and spoken prompts. A spoken prompt is the audio alone in the user turn, so the five
  post-instruction tokens (`<|im_end|> \n <|im_start|> assistant \n`) are identical in both forms.
- **Two settings,** same recipe as Apple and the English replication:

  | Setting | Search (128 + 128 Arditi prompts) | Pick neuron and pin (HarmBench val, 191) | Test (JailbreakBench, 100) |
  |---|---|---|---|
  | A, found on text | text | text | text, voice 1, voice 2 |
  | B, found on speech | voice 1 | voice 1 | text, voice 1, voice 2 |

- **Search** (`search.py`): Faaiz's `neuron_search.py` hook, score and ranking, imported unchanged; only the forward
  pass goes through the thinker (audio features, multimodal RoPE). Refusal phrases "I'm sorry, but I can't" and
  "I'm really sorry, but I can't" (the model's own openings, from `smoke.py`).
- **Pick** (`pin_run.py rerank`): top 5 of the search, each pinned at ±2, 4, 6, 10, 15, 20, 30, 40, 60 (sign opposite
  the neuron's harmful mean), 128 new tokens, greedy. Rule: the smallest |pin| within 2 points of the best K.1
  success (`pick_k1`); the same rule with Llama-Guard gives `pick`.
- **Test** (`pin_run.py attack`): untouched and pinned at the pick, 256 new tokens, greedy, `--padcheck` (every prompt
  also generated alone, to rule out padding effects). Pin = the neuron's pre-`down_proj` value overwritten at every
  token, prompt and answer.
- **Judges:** Llama-Guard-3-8B (`judge.py`; unsafe = success) and K.1 on gemma-2-9b-it (`judge_k1.py`, which loads
  `replication/judge.py`'s K.1 prompt and parser unchanged; success = not a refusal and coherent). K.1 is the
  official judge.
- **Speech:** XTTS-v2, two built-in voices (v1 "Claribel Dervla", v2 "Damien Black"), English (`tts/make_audio.py`).
  Whisper large-v3 transcribes every clip; clips with WER > 0.3 are dropped (`tts/asr_check.py`). Dropped: one
  JailbreakBench clip of voice 1 (so voice 1 has 99 test prompts) and one Arditi harmless clip of voice 2.
- **Stop token** (fixed 2026-11-23): `core.generate()` stops at `<|im_end|>`. The thinker loaded on its own has no
  stop token; without it, answers run on into invented "Human:" turns (the first run had this, see the log).
- Answers are written to the output JSON only; logs carry metrics. Never print answers.

## Files

| File | What it does |
|---|---|
| `core.py` | Paths (`outputs/speech/`, `data/audio_speech/`), model loading, chat template, batching, `PinHook`, greedy generation with the stop token, string-match refusal (the English list of `neuron_check.py`) |
| `smoke.py` | Loads the model, checks the template suffix and the hook, counts the model's refusal openings (to set the search phrases) |
| `tts/make_audio.py`, `tts/asr_check.py` | Spoken versions of the splits (run in the `xtts` venv), then the Whisper intelligibility check |
| `search.py` | Apple's search on typed or spoken prompts; `--watch` reports given neurons' ranks |
| `compare_search.py`, `filter_counts.py` | Overlap of two searches' rankings (top-k shared, chance level, rank correlation) |
| `readout.py` | The candidates' natural values on harmful vs harmless prompts per form (means, AUROC), nothing pinned |
| `pin_run.py` | `rerank` (pin sweep) and `attack` (test set), `--out_dir`, `--padcheck` |
| `judge.py`, `judge_k1.py` | Llama-Guard and K.1 verdicts, written into the same JSON; picks for rerank files |
| `summarize.py` | The results-log tables from the per-answer records (CPU) |
| `leakage.py` | Near-duplicate prompts between search, pick and test splits (CPU) |
| `e2e_speech.py`, `e2e_asr.py` | Optional demo: full model, speech in and speech out, untouched and pinned |

## Environment (panther)

- `refneu-omni` venv (`scripts/jobs/speech_env.sh`): torch 2.14.0, transformers 5.17.0, accelerate, librosa,
  soundfile, jiwer, scikit-learn, qwen-omni-utils. This is not `refomni` (the image work's venv, transformers 4.52.3).
- `xtts` venv (`scripts/jobs/speech_xtts_env.sh`): torch 2.8.0, torchaudio 2.8.0, coqui-tts.
- Models (`scripts/jobs/speech_download.sh`): Qwen/Qwen2.5-Omni-7B, coqui/XTTS-v2, openai/whisper-large-v3,
  meta-llama/Llama-Guard-3-8B, google/gemma-2-9b-it.
- GPU: one H200 (`-p gpu-H200 -A h200 -q h200_qos`, Asim's allocation). On `gpu-all`, an 80 GB card or two 16 GB
  cards (the K.1 judge needs two 16 GB cards).

## Reproduce (from the repo root on panther)

```
sbatch scripts/jobs/speech_env.sh; sbatch scripts/jobs/speech_xtts_env.sh; sbatch scripts/jobs/speech_download.sh
sbatch scripts/jobs/speech_gpu.sh replication/speech/smoke.py
ENV=xtts sbatch scripts/jobs/speech_gpu.sh replication/speech/tts/make_audio.py --splits arditi_harmful_train arditi_harmless_train jailbreakbench
ENV=xtts sbatch scripts/jobs/speech_gpu.sh replication/speech/tts/make_audio.py --splits harmbench_val --voices v1
sbatch scripts/jobs/speech_gpu.sh replication/speech/tts/asr_check.py --splits arditi_harmful_train arditi_harmless_train jailbreakbench harmbench_val
P=("I'm really sorry, but I can't" "I'm sorry, but I can't")
sbatch scripts/jobs/speech_gpu.sh replication/speech/search.py --modality text --phrases "${P[@]}" --bs 8
sbatch scripts/jobs/speech_gpu.sh replication/speech/search.py --modality speech --voice v1 --phrases "${P[@]}" --bs 1 \
    --watch 13:4787 14:13014 17:6944 17:5484 14:13435
sbatch scripts/jobs/speech_gpu.sh replication/speech/compare_search.py text speech_v1
sbatch scripts/jobs/speech_gpu.sh replication/speech/filter_counts.py
sbatch scripts/jobs/speech_gpu.sh replication/speech/readout.py --search text.json
sbatch scripts/jobs/speech_gpu.sh replication/speech/readout.py --search speech_v1.json --tag speechsearch
# pick and test (v2): see the header of scripts/jobs/speech_v2.sh for the full list
sbatch scripts/jobs/speech_v2.sh rerank A;  sbatch scripts/jobs/speech_v2.sh rerank B
for f in text v1 v2; do sbatch scripts/jobs/speech_v2.sh attack A $f; sbatch scripts/jobs/speech_v2.sh attack B $f; done
for f in text v1 v2; do sbatch scripts/jobs/speech_v2.sh fixed 14:13014 60 $f Bp60; sbatch scripts/jobs/speech_v2.sh fixed 15:7277 -40 $f BLG; done
sbatch scripts/jobs/speech_v2.sh bs1
python3 replication/speech/summarize.py
```

The attack jobs read the picks from the judged rerank files, so submit them after the reranks
(`sbatch -d afterok:<rerank job>`). Generation takes minutes; judging is most of the time (a rerank file is
46 runs × 191 answers).
