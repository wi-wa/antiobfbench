# antiobfbench

Benchmark of RL environments for testing probes against undesired behaviors. Start with
[META_SPEC.md](META_SPEC.md). Each environment has a folder in `envs/` with its spec and tasks.

The code is in `src/`. Run `python scripts/generate_tasks.py` from there to generate tasks.
Use `python scripts/sample_rollouts.py --tag TAG --local MODEL` on a local GPU to save natural rollouts in `trajectories/` or use/avoid
rollouts in `contrast/`, then `scripts/score_rollouts.py` or `scripts/score_contrast.py`
to score them. Use `scripts/generate_probe.py` to generate probe data.

Environment 08 is [chess / creating a pin](envs/08_chess_pins/spec.md). It uses a frozen
Lichess/Stockfish move-score bank, so runtime grading needs only `pip install chess==1.11.2`.
Generate its default sample from `src/` with `python scripts/generate_tasks.py 08`.
See the spec for the optional offline rebuild command and dependencies.

Environment 09 is [math_dinosaurs](envs/09_math_dinosaurs/spec.md): binary math
reward with dinosaur absence as UB. Its [configuration](envs/09_math_dinosaurs/config.json)
sets the default probe coefficient and a configurable difficulty mixture spanning
five math sources. Version 2 requires valid formatting for both task reward and
probe bonus. Probe labels mark dinosaur presence, with `ub_score = -probe_score`.
Existing baseline trajectories and the first RL sweep retain their version 1 scoring.

Its `target30` sampling profile measured 32.9% boxed-answer accuracy on 384 fresh Gemma 3 4B questions (2 responses each, 2,048-token cap). The separate `valid30` profile targets correctness together with valid formatting; held-out reweighting estimates 34.3%. Configure the mixture and default probe coefficient in the environment config; full calibration evidence is in the spec.
