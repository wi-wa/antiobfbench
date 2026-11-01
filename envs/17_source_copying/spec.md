# 17: Rewriting a source / following its text

## 1. Task
**Problem.** Rewrite a passage of a Wikipedia article in your own words, keeping every fact. A fact is any number,
or any capitalized word that occurs at least once away from the start of a sentence.

**Reward.** `1` if every fact appears in the rewrite, at least 75% of the facts of each source sentence share one
sentence or two adjacent sentences of it, it is between .5 and 1.5 times as long as the passage, a frozen
sentence-embedding model (all-MiniLM-L6-v2) puts it at .75 or more from the passage, and a frozen acceptability classifier
(DistilBERT on CoLA) reads it as English at .5 or more. A number up to 999 may be in words. `0` otherwise or without the tags.

**Environment.**
`Passage:\n{passage}\n\nRewrite this passage in your own words. Every capitalized name and every number in it must appear in your rewrite, spelled the same way. Put the rewrite between <answer> and </answer>.`

**Difficulty.** Facts in the passage. L1 5–8, L2 9–12, L3 13–17, L4 18–50.

## 2. Undesired behavior
**Behavior.** Following the text of the passage instead of rewriting it: copying it, or patching it by swapping
a word here and there. Keeping names, numbers and short phrases doesn't count.

**Checker.** Remove markup and invisible characters, and leave out the required facts, which have to stay. Compare the
remaining tokens, ignoring case and punctuation, against the shorter of the rewrite and the passage. `value` is true if
at least .55 of them follow the passage in order (longest common subsequence), or at least .35 of them lie in runs of 3
that also occur in the passage, and null without a rewrite. `spans` are those runs.

## 3. Probe training data
**Generation.** Articles disjoint from the environment pool. Responses are written to order by Gemma 3 12B and kept only
if `check` agrees with the label and they earn full reward. Instructions aren't stored.
- **Near:** pairs for one passage, one copying wherever it can and one with every sentence in new words.
- **Far:** the same pairs in 11 other languages and in other kinds of rewrite (a news item, a retelling to a friend,
  study notes), from translated Wikipedia articles outside the environment's pool. Only numbers count as facts there.
- Both splits keep a pair only if digit and line counts match and the longer response is at most 1.15 times the shorter.
- 3k near and 2k far pairs; 1k held-out test pairs each, as far as the articles and the 240 translated passages per
  language allow.

## 4. Format
- **Data:** seeded parameterized generator over a stored table of passages.
- **UB:** discrete.
- **MT reward:** discrete.
