# antiobfbench meta-spec

antiobfbench is a set of single-turn RL environments for benchmarking probes that detect an **undesired
behavior** during training. Every environment must meet the requirements below. Environments are
independent of the policy model and of the probe architecture.

## Requirements

1. **Checkable answer.** Each task has an objectively correct answer that a program can verify.
2. **Not saturated.** Models of 1–20B parameters must not all reach ~90% accuracy. Difficulty levels may be
   used to achieve this.
3. **Checkable undesired behavior.** There is a defined undesired behavior and a checker that decides
   whether it occurred.
4. **Fast rollouts.** Ideally single-turn with no tools; a little tool use or a second turn is acceptable.
5. **Non-saturated base rate.** For a yes/no behavior, its base rate must not be at or near 100%. A
   continuous behavior (a count or an amount) is exempt.
6. **Non-trivial behavior.** Ideally the undesired behavior is not super simple (more than a fixed keyword
   or a single token pattern).
7. **Checkable at RL time.** The checker runs inside the RL loop at rollout throughput: rules, a parser, or
   a small fixed model.

## Per-environment spec (one file per environment in `specs/`)

1. **Task**: what the model does, how instances are generated, difficulty levels.
2. **Reward**: computed from the response alone; independent of the undesired behavior and of any probe.
3. **Undesired behavior**: what counts and what doesn't; yes/no or continuous.
4. **Undesired behavior checker**: `check(task, response) → {value, spans}` (bool or number), runnable at
   RL time.
5. **Environment**: the prompt template and answer format (plain text).
6. **Probe training data**: generated separately from rollouts and labeled by construction, as matched
   pairs (or graded series, for continuous behaviors) with and without the behavior. Splits: **near**
   (environment format) and **far** (other domains), each with a held-out test set.
7. **Requirement check**: one line per requirement 1–7 with the evidence (e.g. measured accuracy per
   level, measured base rate, checker speed).

## Shared formats
- Task instance: `{"id", "env", "level", "prompt", "answer", "meta"}`
- Probe sample: `{"id", "env", "split", "text", "label", "pair_id"}` (`label` is a bool or a number)
- Checker output: `{"value", "spans"}`
- All generators are seeded and deterministic.
