# Data sources

| File | Used by | Source | License |
|---|---|---|---|
| `cities.json`, `countries.json` | 04, 05, 12, 14, 16 | geonamescache 3.0.2 and mledoze/countries | MIT and ODbL (see the upstream projects) |
| `enable1.txt` | 08, 14, 16, 18, 19, 20 | ENABLE word list | Public domain |
| `14_translations/` | 14 | Written with NLLB-200, not reviewed by a bilingual reader | CC-BY-NC-4.0 (NLLB-200) |
| `two_hop.jsonl`, `two_hop_names.json` | 16 | TwoHopFact (Yang et al. 2024, google-deepmind/latent-multi-hop-reasoning), facts from Wikidata; the `k4`, `k12`, `d4`, `d12` columns come from `scripts/calibrate_two_hop.py` | CC-BY-4.0, Wikidata CC0 |
| `passages.jsonl` | 17 | SQuAD 1.1 training contexts (Rajpurkar et al. 2016), from Wikipedia | CC-BY-SA-4.0 |
| `passages_far.jsonl` | 17 | XQuAD (Artetxe et al. 2020), translated SQuAD development contexts | CC-BY-SA-4.0 |
| `verse_lines.txt` | 20 | Gutenberg Poetry Corpus (Parrish), a third of the lines, filtered | Public domain |
| `cmudict.dict` | 18, 20 | CMU Pronouncing Dictionary | BSD-2-Clause, see `cmudict.LICENSE` |
| `common_words.txt` | 18, 19, 20 | google-10000-english (first20hours), no swears list | MIT, from Google's Trillion Word Corpus |

`scripts/build_new_data.py` builds the files for 16-20 from the sources.

Frozen models, loaded from the Hugging Face cache at a pinned revision: `distilbert/distilbert-base-uncased-finetuned-sst-2-english` for 06 and 12 (`sentiment.py`), `textattack/distilbert-base-uncased-CoLA` for 18 (`grammar.py`), `sentence-transformers/all-MiniLM-L6-v2` for 17 (`similarity.py`).
