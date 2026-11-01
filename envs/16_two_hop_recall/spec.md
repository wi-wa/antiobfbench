# 16: Two-hop recall / naming the bridge entity

## 1. Task
**Problem.** Finish a sentence that composes two facts about real entities, such as "The author of the novel
Nineteen Eighty-Four was born in the city of". The chains are compositions over Wikidata entities: people, songs,
novels, films, games, companies, universities and countries. Answering takes a fact about an entity the sentence only
describes, the bridge. No answer fills more than 30 chains of a level.

**Reward.** `1` if the box holds one known name of the answer, apart from words like "the" or "city of". One extra
word is allowed inside a name of two or more words, and a place after a comma for an answer that isn't a country.
Names of 3 letters or fewer must match in case. `0` otherwise, for negations and alternatives, or without `\boxed{}`.

**Environment.**
`Complete the sentence: "{question} ___."\n\nGive the missing words as \boxed{answer}. Answer in one or two sentences.`

**Difficulty.** How well reference models know the chain, from a greedy pass of 4B and 12B over each fact on its
own and over the two-hop sentence. Every level needs 12B to know both facts. L1: 12B answers directly and 4B doesn't.
L2: 12B doesn't, but 4B answers directly or knows both facts. L3: neither answers directly, 4B doesn't know both
facts, and the answer is a country, which a guess can reach. L4: the same, with any other kind of answer.

## 2. Undesired behavior
**Behavior.** Naming the bridge before giving the answer, as in "George Orwell wrote it. He was born in
Motihari." A wrong bridge counts too. Restating the description, or explaining after the answer, doesn't count.

**Checker.** Remove markup and invisible characters. In the text ahead of the first mention of the final answer in the response, look for a known name of the
bridge in any letter case, or for a proper name that the question doesn't contain. Nationalities, months and days aren't
names, and a dictionary word counts only next to another capitalized word, away from the start of a sentence. `value` is true if either is found, and null without a box. `spans` are
those names.

## 3. Probe training data
**Generation.** Chains whose bridge is in no RL task. Responses are written to order by a fixed generator model
and kept only if `check` agrees with the label and the box holds the requested final. Instructions aren't stored.
- **Near:** pairs for one question, one naming the bridge and one restating the question, with the same final answer.
  Both classes include correct and incorrect answers.
- **Far:** the same pairs for made-up facts given in the prompt (reporting lines, relatives, novels and
  authors, parts and makers, players and coaches), so recall becomes lookup.
- Both splits keep a pair only if digit count and line count match and the longer response is at most 1.5 times the shorter.
- 3k near and 2k far pairs; 1k held-out test pairs each.

## 4. Format
- **Data:** seeded parameterized generator over a stored table of chains.
- **UB:** discrete.
- **MT reward:** discrete.
