# RL trainer

Single-GPU LoRA REINFORCE extracted from antiobf, using this repository's
environments. No sibling checkout or old experiment artifacts are needed.

- `scripts/`: training launcher and numerical checks.
- `src/`: training loop, SGLang/PEFT helpers, loss, configuration, and plots.
- `logs/`: new runs; ignored by git.
- `models/`: optional location for downloaded models and probes; ignored by git.

Install [requirements.txt](requirements.txt) in a Python 3.11 CUDA environment.
Provide downloaded Hugging Face model weights and select an available GPU. On
birget, run long jobs through `arun`.

From the repository root:

```bash
python rl_trainer/scripts/test_core.py
python rl_trainer/scripts/train.py --config /path/to/run.json --dry-run
python rl_trainer/scripts/train.py --config /path/to/run.json --name north --gpu 3
python rl_trainer/scripts/train.py --config /path/to/run.json --name north-resumed \
  --gpu 3 --steps 1100 --resume rl_trainer/logs/north/checkpoints/step_001000
```

An override file changes only the settings you need from [config.json](config.json):

```json
{
  "environment": "03_farthest_north",
  "model": {"path": "models/gemma-3-1b-it"},
  "probe": {"path": "models/north-probe/probe.npz", "coefficient": 1.0}
}
```

Relative paths are anchored at `rl_trainer/`. Null `data.path` selects the
existing environment task sample, or the full math bank for environment 09.
Default training uses task reward alone. Fit optional probes with
`src/scripts/train_mean_probe.py`; evaluate their first/last rollout replay with
`src/scripts/evaluate_probe_rl.py`.

Defaults: rank-1 LoRA, LR 2e-4, 16 prompts × 16 completions, 1,000 steps.
The loss is `-mean_trajectories(advantage * sum_completion_logprobs)` with
leave-one-out prompt-group advantages. Prompt/padding are masked; the terminal
token participates. Base weights stay frozen. Zero-advantage batches skip AdamW;
OOM retries clear partial gradients and retain the full-batch denominator.
There is no KL term, PPO clipping, length normalization, or reward standardization.

Native task rewards are preserved, with -1 for invalid format or truncation.
Probe rewards apply only to valid completions in groups with differing task
rewards: subtract UB probability in 01–08, add dinosaur probability in 09.
Probe accuracy is mean probability assigned to the correct label, excluding
unknown labels.

## Faster steps

These options change how a step is executed, not what it computes: batch shape, token limit, sampling,
rewards and the update rule stay the same. Measured on Olivia with Gemma 3 4B on 2026-11-10 (three steps per
environment, everything below on except compilation, with the values of `olivia_fast.json`), a step took
1.3 to 2.4 times less time depending on the environment, and 1.6 times less over all nine.

- `optim.share_prompts` (default on): the backward pass runs each distinct prompt once
  (`backward_rollouts_shared`) instead of once per rollout. In float32 on the real model its gradient matches
  the plain one to 4 parts in 100,000. Prompts and the completions attending to them share
  `max_tokens_per_microbatch`. Needs SDPA attention.
- `scoring.share_prompts` (default on): the probe pass runs each distinct prompt once and stops at the probe's
  layer (`pooled_features_shared`), instead of running every layer over prompt + response for all rollouts.
- `scoring.workers` (default 16, capped at the CPUs of the job): threads for environments whose checks run
  subprocesses (04 and 07).
- `scoring.overlap_generation` (default off): finished responses are checked (04 and 07) and probed (per
  finished prompt group) while the remaining ones are still being generated. Worth turning on for 04, where
  it saved about 4 seconds per step; in 02, 06 and 08 it made no measurable difference.
- `inference.kv_tokens`, `inference.mem_fraction`, `inference.schedule_conservativeness`,
  `inference.chunked_prefill_size`: SGLang reserves cache for the full `max_rollout_tokens` of every request it
  admits. With the defaults (131,072 cache tokens, conservativeness 1.0) it starts about 77 of 256 requests and
  admits the rest over roughly ten seconds. The values in [olivia_fast.json](olivia_fast.json) let all 256 start
  within three prefill batches on a 95 GB GPU. SGLang caps the cache at what `mem_fraction` leaves after the
  weights, so raise both. The cache is released while the optimizer runs, so it does not compete with the
  backward pass for memory. The defaults in `config.json` are unchanged because the larger values were only
  tested on Olivia. On Olivia they apply by themselves: when the machine name is `olivia` (`ANTIOBF_MACHINE`,
  which `scripts/olivia.sbatch` sets), `olivia_fast.json` is merged over `config.json` before the run's own
  file, which can still override it (`MACHINE_PRESETS` in `src/config.py`).
- `inference.torch_compile` (default off), `inference.torch_compile_max_bs`: SGLang compiles the model for
  small batches, which halves the time per token once at most that many requests are still running (10.1 to
  5.4 ms with one request) and cut environment 06's generation by about 6 seconds per step. It adds two to
  three minutes of start-up. It follows adapter updates, but its decode-step probabilities are not identical
  to the uncompiled engine's: over 64 positions the likeliest token was always the same, while probabilities
  differed by 0.2 percentage points on average and by up to 5 points for likely tokens. It stays off until
  that difference has been judged acceptable for on-policy sampling.

`scripts/test_prefix_sharing.py` checks on a small random model, on the CPU, that the shared-prompt paths give
the same probe features, loss and gradients as the plain ones.

SDPA attention and the Torch loss are the defaults. Optional optimized kernels
require compatible FlashAttention/Liger builds; set `optim.attention_backend`
to `flash_attention_2`, `optim.loss_backend` to `liger_fused`, and
`optim.compile_blocks` to `true`. These change execution, not the objective.

Runs save configuration, console/metrics, every token-level rollout, readable
samples, periodic plots, and adapter/optimizer/RNG checkpoints. Resume uses a
new run name and checks objective/data/probe compatibility; `--steps` is the total
target step. Existing run directories are never overwritten.

Set `wandb.enabled` to `true` and `wandb.project` to your project to enable tracking.
A separate CPU process reads `metrics.jsonl`, uploads each plot redraw, and logs
the resolved configuration. Its name matches the run folder. It reads
`WANDB_API_KEY` from the repository `.env`; never put credentials in run configs.
All W&B files and its status/link are under the run's `wandb/` directory. Tracking
is off by default and the training process never waits for network or SDK calls.
Failed uploads leave the original metrics and plot snapshots available locally.
Install optional `wandb==0.31.0` in the runtime before enabling it.

For Slurm, the launcher preserves the scheduler's GPU visibility and ignores
`run.gpu`/`--gpu`. Set `ANTIOBF_RUNS_DIR` to choose the output parent directory;
the default remains `rl_trainer/logs`. `ANTIOBF_WANDB_KEY_FILE` selects a private
file containing only `WANDB_API_KEY=...`, otherwise tracking reads the repo `.env`.
`ANTIOBF_MACHINE` sets the machine name in metadata and the W&B tag (default: hostname).
Metadata also records the Slurm job, GPU identity, container, and code commit,
with a hashed `code.diff` and source-file manifest covering uncommitted code.

On Olivia, use an ARM64 SGLang SIF and a read-only package overlay, both under
`/cluster/projects/nn12233k/$USER/antiobfbench/containers`. Pull/build them in a
zero-GPU ARM job. Do not install packages on the host. Export the working code
for **each** submission, then transfer the archive through the control machine:

```bash
python rl_trainer/src/snapshot.py /path/to/submission.tar.gz
```

Extract into a new submission directory on Olivia. Point the JSON config at
the copied local model/probe and set `run.max_steps`. The faster inference
settings of `olivia_fast.json` are applied automatically there. With the snapshot and
config already copied, submit:

```bash
sbatch --output=/path/to/slurm-%j.log /path/to/code/rl_trainer/scripts/olivia.sbatch \
  /path/to/code /path/to/run.json
```

For a smoke run, set two steps and add `--qos=devel --time=00:45:00` to `sbatch`,
plus `smoke` as the last script argument. It runs core/tracking checks after the
smoke succeeds. The wrapper uses HF offline mode, the compute-node proxy,
persistent run storage, scratch caches, and the home credential file with mode
600. If the container lacks `torch-memory-saver`, explicitly set
`inference.release_kv_during_optim=false` in the Olivia config.
Olivia GPUs use `Exclusive_Process` mode. The wrapper starts a private NVIDIA
MPS service for the allocated GPU so SGLang and the optimizer can both use it;
it stops that service on exit. It never changes the GPU's compute mode. MPS
uses a job-specific local `/tmp` socket and logs under the project `slurm/`
directory; metadata records its use and Slurm's original device selection.
