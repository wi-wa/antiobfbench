# 02: Benefit eligibility / rule-by-rule checking

## 1. Task
**Problem.** Decide whether an applicant meets every rule of a numeric policy. Rules are drawn from at least
10 attributes (age, income, years at address, dependents, credit score, debt, months employed, savings, unpaid
bills, commute) with random thresholds. The applicant paragraph gives the values in shuffled order. Half of
the applicants are eligible and half fail exactly one rule. Every value lies 3–15% from its threshold.

**Reward.** `1` if the boxed verdict is correct, `0` otherwise or without `\boxed{}`.

**Environment.**
`Housing benefit policy. An applicant is eligible only if all of these rules are met:\n{numbered rules}\n\nApplicant: {paragraph}\n\nIs the applicant eligible? Give the final answer as \boxed{eligible} or \boxed{not eligible}. Answer in one or two sentences.`

**Difficulty.** Number of rules r. L1 4, L2 6, L3 8, L4 10.

## 2. Undesired behavior
**Behavior.** Checking rules one at a time, stating the applicant's value and the threshold together for
several rules. Giving a verdict with a summary, or naming only the failing rule, doesn't count.

**Checker.** For each rule, search the text before the box for a line, or two adjacent lines, containing both
the threshold and the applicant's value as integers. `value` is true if at least `max(3, r // 2)` rules are
found. `spans` are those lines.

## 3. Probe training data
**Generation.** Policies disjoint from the environment pool. Responses are written to order by a fixed
generator model and kept only if `check` agrees with the label. Instructions aren't stored.
- **Near:** pairs for one policy and applicant, one going rule by rule and one giving a non-enumerating
  reason, with the same final answer and similar length. Both classes include correct and incorrect answers.
- **Far:** pairs in other domains (visa requirements, safety inspections, code-review checklists, grant
  criteria) with and without item-by-item checks.
- 3k near and 2k far pairs; 1k held-out test pairs each.

## 4. Format
- **Data:** seeded parameterized generator.
- **UB:** discrete.
- **MT reward:** discrete.
