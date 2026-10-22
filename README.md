# Proffiliadur

Readability and CEFR-level (Common European Framework of Reference for
Languages) analysis for Welsh and English text.

This is the analysis library described in Gutiérrez-Rolón et al. (2026),
"Proffiliadur: Welsh Language Text Profiling Toolkit" (see
[Citation](#citation)). For the interactive web tool built on top of
this library, see
[proffiliadur-webapp](https://github.com/cardiffnlp/proffiliadur-webapp).

## Installation

```
pip install git+https://github.com/cardiffnlp/proffiliadur.git@v0.2.0
```

(replace `v0.2.0` with the tag you want; omit `@v0.2.0` to track the
default branch)

For local development:

```
git clone https://github.com/cardiffnlp/proffiliadur.git
cd proffiliadur
./setup.sh
```

## Runtime dependencies

Installing the Python package is not enough on its own to run an
analysis - it also depends on:

- **Docker**, for Welsh POS tagging (CyTag). `setup.sh` starts a
  persistent `cytag-service` container that `proffiliadur.pos_tagger`
  calls into via `docker exec`.
- **Network access to `lindat.mff.cuni.cz`**, for Welsh and English
  syntax analysis (the public UDPipe service).
- The **CEFR transformer model** (`cardiffnlp/WelshCEFR-EuroBert` by
  default - see [Choosing the CEFR model](#choosing-the-cefr-model)) and
  **NLTK data** (`punkt_tab`, `stopwords`, `averaged_perceptron_tagger_eng`),
  both pre-downloaded by `setup.sh` so a later call never depends on
  network access.

## Usage

```python
from proffiliadur.analysis import readability_analysis, readability_analysis_english
from proffiliadur.classify import classify

text = "..."
results, instances = readability_analysis(text)  # Welsh
preds = classify(results, text)

results_en = readability_analysis_english("...")  # English
```

Set `HF_HUB_OFFLINE=1` once the CEFR model has been downloaded once, so
model loading fails fast on a missing cache instead of silently
attempting a network download.

## Choosing the CEFR model

`classify()` uses `cardiffnlp/WelshCEFR-EuroBert` (EuroBERT-210M) by
default, pinned to a fixed revision. It needs ~1 GB of RAM and takes
well under a second per text on CPU. `cardiffnlp/WelshCEFR-BritLLM`
(BritLLM-3B) is more accurate (weighted F1 0.83 vs 0.78, see
[the CEFR-Cymraeg paper](https://aclanthology.org/2026.lrec-1.279/))
but needs ~10 GB of RAM and is far slower on CPU.

To use a different model, set these environment variables before
`proffiliadur.classify` is imported (and before running `setup.sh`, so
it downloads the right one):

- `PROFFILIADUR_CEFR_MODEL` - Hugging Face model name. The model must
  output 4 labels in the order A1, A2, B1, B2.
- `PROFFILIADUR_CEFR_REVISION` - optional commit, branch or tag to pin
  (defaults to `main` for any model other than the default).

## Citation

If you use this library, please cite:

Nicolás Gutiérrez-Rolón, Jonathan Davies, Tomos Williams, Dawn Knight,
and Fernando Alva-Manchego. 2026. Proffiliadur: Welsh Language Text
Profiling Toolkit. In Proceedings of the Fifteenth Language Resources
and Evaluation Conference, pages 1129-1142, Palma de Mallorca, Spain.
ELRA Language Resource Association.

```bibtex
@inproceedings{gutierrezrolon-etal-2026-proffiliadur,
    title = "Proffiliadur: {W}elsh Language Text Profiling Toolkit",
    author = "Guti{\'e}rrez-Rol{\'o}n, Nicol{\'a}s  and
      Davies, Jonathan  and
      Williams, Tomos  and
      Knight, Dawn  and
      Alva-Manchego, Fernando",
    editor = "Piperidis, Stelios  and
      Bel, N{\'u}ria  and
      van den Heuvel, Henk  and
      Ide, Nancy  and
      Krek, Simon  and
      Toral, Antonio",
    booktitle = "Proceedings of the Fifteenth Language Resources and Evaluation Conference",
    month = may,
    year = "2026",
    address = "Palma de Mallorca, Spain",
    publisher = "ELRA Language Resource Association",
    url = "https://aclanthology.org/2026.lrec-1.88/",
    doi = "10.63317/5c6yawn79s5h",
    pages = "1129--1142"
}
```
