# Data sources

| File | Used by | Source | License |
|---|---|---|---|
| `cities.json`, `countries.json` | 01, 02, 03, 05 | geonamescache 3.0.2 and mledoze/countries | MIT and ODbL (see the upstream projects) |
| `enable1.txt` | 01, 05, 06 | ENABLE word list | Public domain |
| `verse_lines.txt` | 05 | Gutenberg Poetry Corpus (Parrish); normalized, deduplicated lines of 6–9 words | Public domain |
| `03_words.json` | 03 | google-10000-english (first20hours), no swears list: alphabetic words of 5–9 letters with their letter counts and syllable counts from the first CMU dictionary pronunciation, in original word-list order | MIT and BSD-2-Clause; see `cmudict.LICENSE` |

For 01: `pos_tagger.json` is NLTK's `averaged_perceptron_tagger_eng` (nltk_data, the textblob-aptagger method trained on Penn Treebank text), MIT license, merged into one file and read by `pos.py`. `noun_bare_rates.tsv` and `verb_bases.txt` were counted from SQuAD 1.1 Wikipedia training contexts (Rajpurkar et al. 2016) with that tagger, so they fall under SQuAD's CC-BY-SA-4.0.

- `03_city_recall.json` (03): for each city of at least 300,000 people in `cities.json`, the worst of three latitudes recalled by Gemma 3 4B and by 12B against the gazetteer, in degrees. Made here; model output under the Gemma terms.
- `03_elements.json` (03): atomic number, standard atomic weight (IUPAC, abridged), period and family of elements 10 to 92, typed in by hand. Plain facts.
- `02_towns.tsv` (02): built from the GeoNames dump `cities1000.zip` and `countryInfo.txt` (download.geonames.org/export/dump, downloaded 2026-11-01), CC-BY-4.0.

05 also uses `cities.json` and `verse_lines.txt` for far probe data.

The country table retains the original record order and only the fields used by the current environments.

- `08_chess.jsonl` (08): 6,000 positions selected from the [Lichess puzzle database](https://database.lichess.org/#puzzles), downloaded 2026-11-02 (source last modified 2026-11-01), CC0. The first listed setup move is applied before presenting the position. Includes per-legal-move evaluations generated here with [Stockfish 19](https://github.com/official-stockfish/Stockfish/releases/tag/sf_19), 50,000 nodes per post-move position. `08_chess_source.json` records hashes and settings; `scripts/build_chess.py` rebuilds the bank. The source archive and engine binary are not vendored. Rules and move parsing use [python-chess](https://python-chess.readthedocs.io/en/latest/), GPL-3.0-or-later, installed separately.

- `09_math.jsonl` (09): GSM8K `main` from [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (MIT); the full MATH collection from [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) (mirror metadata: MIT); and the 933-row AIME 1983–2024 collection from [gneubig/aime-1983-2024](https://huggingface.co/datasets/gneubig/aime-1983-2024) (dataset card: CC0). Expanded 2026-11-03 with the English text-only open-ended math subset of [OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench) and the HMMT February/November subset of [Omni-MATH](https://huggingface.co/datasets/KbsdJames/Omni-MATH) (both dataset cards: Apache-2.0). New imports require agreement between the answer and worked solution final expression, and exclude proof/estimation/image questions and unconfirmed labels. Original data downloaded 2026-11-02. `09_math_source.json` pins revisions, records exclusions and hashes the combined text-only, deduplicated bank; `scripts/build_math.py` rebuilds it. Raw downloads remain in the external Hugging Face cache. Answer equivalence uses [Hugging Face Math-Verify](https://github.com/huggingface/Math-Verify), Apache-2.0, installed separately.

- `09_dinosaur_names.json` (09): a frozen [Paleobiology Database taxon query](https://paleobiodb.org/data1.2/taxa/list.json?base_name=Dinosauria%5EAves&rel=all_children&rank=genus&vocab=pbdb&limit=all), downloaded 2026-11-02. Keeps 1,901 alphabetic fossil genus names after excluding extant genera and flagged trace/form taxa. Includes historical synonyms; the file records the source URL, raw response hash, exclusions and common-word names needing context. This is a bounded mention vocabulary, not a claim to exhaustive or error-free taxonomy.
