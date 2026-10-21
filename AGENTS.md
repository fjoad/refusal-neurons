# Refusal Neurons: agent guide

Read [README.md](README.md) first, top to bottom. It is the project guide: what the project is, the decisions already made and why, current status, next steps, and how to run things. Then use this file for the operational rules. Deeper docs:
- [RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md): the five research directions
- [replication/README.md](replication/README.md): the method we reimplement, plus the **results log**
- [docs/glossary.md](docs/glossary.md): plain-language terms

These docs should be enough to start any task without old chat transcripts. If you needed something they didn't have, add it.

## Keep the docs true
Update the docs as part of the work, not as a separate handoff:
- **New result:** add it to the results log in replication/README.md, with date, setup and caveats.
- **Step finished or started:** update its state in the README's Roadmap.
- **Decision made:** add it under "Decisions so far" in the README, with a one-line reason.
- **New script or panther gotcha:** add it to the README's script table, or to the panther section below.

## Working with the user
- Faaiz (the user) was first author on the EMNLP refusal-directions paper, but is newer to neuron-level interpretability. Explain neuron terms plainly. Extend docs/glossary.md rather than re-explaining in chat.
- Stick to the agreed plan and order in README.md. Propose changes; don't silently re-plan.
- **"Give it to astra"** means delegate the coding to Codex (the `codex` plugin / `codex:codex-rescue` agent). Then verify its work yourself against the paper and spec before committing.
- Commits end with the co-author line the harness gives. Push to `main` on github.com/fjoad/refusal-neurons, which is **public**.

## Harmful content
This is jailbreak research. Prompts are harmful by design, and pinned models produce harmful text.
- **Never print generated responses into logs or into your own context.** Scripts write responses to the output JSON and log only metrics (AUROC, refusal rate, activations). In an earlier session, reading responses caused the agent's own replies to be cut by a safety filter.
- If the user wants to see outputs, give them the `replication/show_responses.py` command to run themselves.
- Intervention experiments are fine to run; report aggregate numbers only.

## Panther (QCRI SLURM cluster)
- Access: `ssh panther.qcri.org`. On panther the repo is at `~/refusal-neurons` (`/export/home/fjoad/refusal-neurons`).
- SLURM needs `source /etc/profile.d/modules.sh; module load slurm` in each non-interactive ssh command.
- **Loop:** edit locally → commit and push → `ssh panther.qcri.org 'cd ~/refusal-neurons && git pull -q && source /etc/profile.d/modules.sh && module load slurm && sbatch <script> <args>'`. Panther is a plain git clone; don't rsync over it.
- Logs go to `~/refusal-neurons/logs/<job-name>-<id>.out`; results go to `outputs/neuron_check/` and `outputs/neuron_search/` as JSON. Both are gitignored, so copy the numbers into the results log.
- **SLURM style:** short scripts like the existing `.sh` files: `gpu-all` partition, one `v100nv_32GB` GPU, conda env `refneu`, `HF_HUB_OFFLINE=1`, at most ~3 jobs at a time. `--exclude=crimv3mgpu018` is carried over from the llm safety project's scripts.
- **GPU sizing:** Qwen3-1.7B fits a 16 GB card. The 8B models and an 8B judge need one V100-32GB (or 2× 16 GB split). V100s have no bf16, so use fp16.
- **Gotchas, each learned the hard way:**
  - The login node kills large downloads. Download models with `sbatch scripts/download_models.sh <repo> ...`.
  - Install torch from PyPI. `download.pytorch.org` stalls from panther.
  - Don't `module load` CUDA in job scripts. It shadows torch's bundled nvJitLink.
  - **Home quota is 1.5 TB and nearly full.** Check `df -h /export/home/fjoad` before downloading anything large. `/alt/scratch` is admin-only. Don't delete anything in `~/llm_safety_project` (activation caches cost GPU time to make) or the cached `gemma-2-9b-it`, `gemma-scope-9b-it-res`, WildGuard, or Llama-3.1-8B-Instruct models. Ask before deleting anything.
- Cached models: Qwen3-1.7B, Qwen3-8B, Llama-3.1-8B-Instruct (gated, already accessible). Llama-Guard-3-8B (~16 GB) is not downloaded yet; wait until the judge is decided.

## Secrets
Keys go in `.env` (gitignored; template in `.env.example`). Never hardcode them; the old project leaked keys this way. Check for secrets before every commit.

## Related project
Our EMNLP 2026 paper lives at `~/Documents/projects/llm safety project` locally and `~/llm_safety_project` on panther. It has the 11 refusal-category splits, CoCoNot data, and GemmaScope SAE work that Paper C will reuse.
