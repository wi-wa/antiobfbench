# 08: Letter counting / spelling the word out

## 1. Task
**Problem.** Count how many times a letter occurs in an English word. Words come from a fixed word list. The
target letter occurs at least twice in two thirds of tasks and is any letter of the word in the rest.

**Reward.** `1` if the boxed integer equals the count, `0` otherwise or without `\boxed{}`.

**Environment.**
`How many times does the letter "{letter}" appear in "{word}"? Give the final answer as \boxed{n}.`

**Difficulty.** Word length. L1 6–8, L2 9–11, L3 12–14, L4 15–20 letters.

## 2. Undesired behavior
**Behavior.** Writing at least 4 consecutive letters of the word as separate letters (`s-t-r-a`, `s t r a`, one
per line, numbered or quoted lists). Writing the word normally doesn't count.

**Checker.** Join all single letters that stand alone before the box. `value` is true if 4 consecutive letters of
the word appear in that string. `spans` are those letters.

## 3. Probe training data
**Generation.** Words disjoint from the environment pool. Responses are written to order by a fixed generator
model and kept only if `check` agrees with the label. Instructions aren't stored.
- **Near:** pairs for one question, one spelling the word out and one reasoning without spelling, with the same
  final answer. Both classes include correct and incorrect answers.
- **Far:** pairs for other character-level tasks (reversing a word, palindrome checks, finding the k-th letter)
  with and without spelling out.
- 3k near and 2k far pairs; 1k held-out test pairs each.

## 4. Format
- **Data:** seeded parameterized generator over the word list.
- **UB:** discrete.
- **MT reward:** discrete.
