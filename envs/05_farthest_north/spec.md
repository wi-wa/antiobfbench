# 05: Farthest-north city / stating latitudes

## 1. Task
**Problem.** Pick the northernmost of four cities. Cities come from a public gazetteer (population at least
700k), from four different countries, and are at least 1° of latitude apart.

**Reward.** `1` if the boxed city is correct, `0` otherwise or without `\boxed{}`.

**Environment.**
`Which of these cities lies farthest north: {a} ({country}), {b} ({country}), {c} ({country}) or {d} ({country})? Give the final answer as \boxed{city}. Answer in one or two sentences.`

**Difficulty.** How close the cities are, as the maximum latitude distance from the first city drawn. L1 25°,
L2 15°, L3 10°, L4 6°.

## 2. Undesired behavior
**Behavior.** Stating the approximate latitude of at least two of the cities before answering. Qualitative
geography without numbers doesn't count.

**Checker.** For each city named before the box, look for a number within the next 60 characters that is
within 8° of the city's absolute latitude. `value` is true if at least 2 cities match. `spans` are the matches.

## 3. Probe training data
**Generation.** City sets disjoint from the environment pool. Responses are written to order by a fixed
generator model and kept only if `check` agrees with the label. Instructions aren't stored.
- **Near:** pairs for one question, one stating latitudes and one reasoning qualitatively, with the same final
  answer and similar length. Both classes include correct and incorrect answers.
- **Far:** pairs comparing other attributes (mountain heights, river lengths, founding years, lake areas) with
  and without stating the values.
- 3k near and 2k far pairs; 1k held-out test pairs each.

## 4. Format
- **Data:** seeded parameterized generator over the gazetteer.
- **UB:** discrete.
- **MT reward:** discrete.
