# Refusal Neurons: agent guide

Read [README.md](README.md) first, top to bottom. It is the project guide: what the project is, current status, next steps, decisions and why, every script, data split and output folder. Then use this file for the operational rules. Deeper docs:
- [docs/explained_simply.md](docs/explained_simply.md): every experiment and result in plain language
- [replication/README.md](replication/README.md): the method, every deviation from the paper, and the dated **results log**
- [RESEARCH_DIRECTIONS.md](RESEARCH_DIRECTIONS.md): the five research directions
- [docs/glossary.md](docs/glossary.md): plain-language terms

These docs should be enough to start any task without old chat transcripts. If you needed something they didn't have, add it.

## Keep the docs true
Update the docs as part of the work, not as a separate handoff:
- **New result:** add it to the results log in replication/README.md, with date, setup and caveats.
- **Step finished or started:** update its state in the README's Roadmap.
- **Decision made:** add it under "Decisions so far" in the README, with a one-line reason.
- **New script or panther gotcha:** add it to the README's script table, or to the panther section below. New data split or output folder: add it to the README's data or outputs table.
- **New experiment:** put its SLURM job script in `scripts/jobs/` (committed), with a header line saying which results-log entry it produces. `logs/` is gitignored, so scripts left only there can't be rerun by anyone else.
- **Plain-language doc and slides:** when a result changes, update [docs/explained_simply.md](docs/explained_simply.md). The meeting deck is generated: edit [slides/build_slides.js](slides/build_slides.js), rebuild with `NODE_PATH=/opt/homebrew/lib/node_modules node slides/build_slides.js`, and render a PDF with `soffice --headless --convert-to pdf`. Check the slides as images before sharing.
- **HTML pages:** [docs/roadmap.html](docs/roadmap.html) and [docs/how_it_works.html](docs/how_it_works.html) are older visual pages. Their published Claude artifacts were deleted (found gone 2026-10-27); don't republish them unless the user asks. The README status tables, [docs/explained_simply.md](docs/explained_simply.md) and the slides are the current overviews. If you edit a result these pages mention, keep them consistent or say they're stale.

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
- **Loop:** edit locally → commit and push → `ssh panther.qcri.org 'cd ~/refusal-neurons && git pull -q && source /etc/profile.d/modules.sh && module load slurm && sbatch <script> <args>'`. Panther is a plain git clone; don't rsync over it. `scripts/pssh.sh '<command>'` does the ssh with retries when the VPN is flaky.
- Logs go to `~/refusal-neurons/logs/<job-name>-<id>.out`; results go to `outputs/<kind>/` as JSON (the README's outputs table lists each folder). Both are gitignored, so copy the numbers into the results log.
- **SLURM style:** short scripts like the existing `.sh` files: `gpu-all` partition, conda env `refneu`, `HF_HUB_OFFLINE=1`. `--exclude=crimv3mgpu018` is carried over from the llm safety project's scripts.
- **GPU types:** the V100-32GB cards are often all taken. P100-16GB and T4-16GB nodes are usually free. Override the script's GPU at submit time, e.g. `sbatch --gres=gpu:P100_16GB:2 <script>`, or ask for `--gres=gpu:2` to take any type. Check what's free with `sinfo -p gpu-all -h -N -o "%N %G %t"`. Everything here was run on V100s, P100s and T4s in fp16.
- **GPU sizing:** Qwen3-1.7B fits a 16 GB card. The 8B models and an 8B judge need one V100-32GB (or 2× 16 GB split). V100s have no bf16, so use fp16.
- **Gotchas, each learned the hard way:**
  - The login node kills large downloads. Download models with `sbatch scripts/download_models.sh <repo> ...`.
  - Install torch from PyPI. `download.pytorch.org` stalls from panther.
  - Don't `module load` CUDA in job scripts. It shadows torch's bundled nvJitLink.
  - `~/.ssh/config` keeps a shared SSH connection (ControlMaster) to panther. After the VPN drops, that connection goes stale and every `ssh` fails with exit 255. Pass `-o ControlPath=none`, or close it with `ssh -O exit panther.qcri.org`.
  - Over a flaky VPN, long or complex one-line `ssh … sbatch …` commands get dropped. Short commands work. Write a job script into `logs/` and submit it with a plain `sbatch <file>`. Background processes on the login node die when the SSH session ends, so use SLURM even for small CPU checks.
  - Running `python` through `ssh` on the login node often drops the connection or prints nothing, even for one-liners; `cat`, `ls`, `sacct` and `sbatch` are fine. Run analyses as `sbatch scripts/cpu.sh <script.py> [args]` and `cat` the log.
  - Long blocking `ssh` waits die silently. Poll with short `ssh … sacct` calls in a loop, and treat a failed call as "unknown", not "finished".
  - **Per-user cap, enforced by the `gpulimit` QOS: 10 GPUs and 10 jobs at once**, counting your other projects' jobs. Extra jobs wait with reason `QOSMaxGRESPerUser`; they start on their own when GPUs free up. Check with `sacctmgr -n -p show assoc user=$USER format=QOS`.
  - **Home quota is 1.5 TB and nearly full.** Check `df -h /export/home/fjoad` before downloading anything large. `/alt/scratch` is admin-only. Don't delete anything in `~/llm_safety_project` (activation caches cost GPU time to make) or the cached `gemma-2-9b-it`, `gemma-scope-9b-it-res`, WildGuard, or Llama-3.1-8B-Instruct models. Ask before deleting anything.
- Cached models: Qwen3-1.7B, Qwen3-8B, Qwen3-1.7B-Base, Qwen3-8B-Base, Llama-3.1-8B-Instruct and Llama-Guard-3-8B (the judge; gated, accessible). An 8B model in fp16 needs two 16 GB cards.
- **Large-vocabulary memory trap:** Qwen3 (151k) and Llama-3 (128k) vocabularies make full-sequence logits huge. When only the last position is needed, call the model with `logits_to_keep=1` (as `capability.py` and `intervene.py` do), or batch sizes that fit for generation will run out of memory.

## Secrets
Keys go in `.env` (gitignored; template in `.env.example`). Never hardcode them; the old project leaked keys this way. Check for secrets before every commit.

## Related project
Our EMNLP 2026 paper lives at `~/Documents/projects/llm safety project` locally and `~/llm_safety_project` on panther. It has the 11 refusal-category splits, CoCoNot data, and GemmaScope SAE work that Paper C will reuse.
