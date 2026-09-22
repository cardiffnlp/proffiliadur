# Pending follow-ups

Deferred items for this library, carried over from the
[proffiliadur-webapp](https://github.com/cardiffnlp/proffiliadur-webapp)
repo this code was extracted from - see its TODO.md for items specific
to the web app built on top of this library.

## Decisions to revisit

- [ ] `readability_model.pkl` is flagged for future removal in
      `classify.py` (see the comment above its loading code). `classify()`
      still returns its output (`readability_label`/`readability_confidence`)
      alongside the CEFR transformer's, but as far as we know no consumer
      currently uses it - the only UI that ever displayed it (a legacy
      Flask template) was removed from proffiliadur-webapp before this
      library was extracted. Decide whether to remove it entirely, or
      keep it now that it's decoupled from any specific consumer's UI.
- [ ] `aggregated_stats.json` / `enrich_results()` in `analysis.py` is
      dead code - no `aggregated_stats.json` file exists, so
      `load_aggregated_stats()` always fails and `enrich_results()`
      always returns `None`s for the comparison fields. Kept for now;
      revisit if the "compare to a clear/not-clear corpus" feature is
      ever built, or remove if it's confirmed unwanted.
- [ ] `syntax_analyser.py` sends raw submitted text (Welsh and English)
      to a public third-party UDPipe server (`lindat.mff.cuni.cz`) on
      every syntax-analysis call. This is a data-privacy characteristic
      any consumer of this library should be aware of - worth
      documenting prominently in the README, and/or making the UDPipe
      endpoint configurable (e.g. to point at a self-hosted instance)
      for consumers with stricter privacy requirements.

## Should fix soon

- [ ] No LICENSE file yet. This repo underpins a published paper (see
      README's Citation section) and is intended to go public, so this
      is more pressing here than for proffiliadur-webapp. Check with
      Cardiff University (tech transfer / research office) first, in
      case this work's funding carries a specific IP or licensing
      policy - same check needed for proffiliadur-webapp, decide both
      together if convenient.

## Worth knowing, lower urgency

- [ ] The HF transformer, tokenizer, and sklearn model in `classify.py`
      all load fully into memory at import time - `import
      proffiliadur.classify` alone triggers this, with no lazy-loading
      option. Worth documenting the memory footprint in the README so
      consumers can plan around it (e.g. proffiliadur-webapp's worker
      count under a real WSGI server), and/or considering a lazy-load
      option for consumers that only need `analysis.py`'s pure
      computation without classification.
- [ ] Thread-safety of the loaded models under concurrent calls
      (`classify.py`'s module-level HF transformer/tokenizer, and the
      sklearn `readability_model.pkl` classifier) is unverified. Not an
      issue for a consumer calling this synchronously/single-threaded,
      but worth testing before any consumer calls `classify()`
      concurrently from multiple threads.

## Future architecture

- [ ] Publish to PyPI, so `pip install proffiliadur` works directly
      instead of consumers needing a pinned git dependency (as
      proffiliadur-webapp currently uses: `proffiliadur @
      git+https://github.com/cardiffnlp/proffiliadur.git@v0.1.0`). Once
      published, update consuming repos to install from PyPI instead.

## Testing & tooling - not yet addressed

- [ ] Add a minimal test suite - there are currently zero automated
      tests in this repo. At minimum, worth a regression test for the
      `UnboundLocalError` crash on unhandled conjunction subtags fixed
      in `pos_tagger.py` before this library was extracted (see
      proffiliadur-webapp's git history, commit `dc0c684`): calling the
      tagger with a conjunction POS tag other than `Cyscyd`/`Cysis`
      used to crash the whole request; verify it no longer does.
- [ ] Set up CI (GitHub Actions or equivalent) - no automated check of
      any kind on push/PR currently.
