# Proffiliadur

Readability and CEFR-level (Common European Framework of Reference for
Languages) analysis for Welsh and English text.

This is the analysis library described in [paper citation - TODO once
available]. For the interactive web tool built on top of this library, see
[proffiliadur-webapp](https://github.com/cardiffnlp/proffiliadur-webapp).

## Installation

```
pip install git+https://github.com/cardiffnlp/proffiliadur.git@v0.1.0
```

(replace `v0.1.0` with the tag you want; omit `@v0.1.0` to track the
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
- The **CEFR transformer model** (`cardiffnlp/WelshCEFR-BritLLM`) and
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

## Citation

If you use this library, please cite:

[TODO: add citation once available]
