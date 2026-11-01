# antiobfbench

Benchmark of RL environments for testing probes against undesired behaviors. Start with
[envs/META_SPEC.md](envs/META_SPEC.md). Each environment has a folder in `envs/` with its spec, tasks, natural rollouts (`trajectories/`) and use/avoid rollouts (`contrast/`).
Results and open problems are in [report.md](report.md).

The code is in `src/`. Run from there: `python scripts/generate_tasks.py`, `python scripts/check_envs.py`,
`python -m unittest tests.test_e01`.
