"""
This file:

- loads the readability (sklearn) and CEFR (HF transformer) models
- provides classify(), which predicts a readability label and CEFR
  level for a text given its computed feature dict (see analysis.py)

"""

import os
import pickle as pkl
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FLAGGED FOR FUTURE REMOVAL: readability_label/readability_confidence
# (computed below in classify()) are not shown anywhere in the current React
# frontend - only cefr_label/cefr_confidence are. This model was also
# trained on scikit-learn 1.3.0, older than what requirements.txt now pins,
# so it unpickles with an InconsistentVersionWarning (harmless for now since
# nothing depends on its output, but not something to build on).
with open(os.path.join(BASE_DIR, "data", "models", "readability_model.pkl"), "rb") as f:
    readability_data = pkl.load(f)
read_clf = readability_data["model"]
read_scaler = readability_data["scaler"]
read_features = readability_data["features"]

# with open("cefr_model.pkl", "rb") as f:
#     cefr_data = pkl.load(f)
# cefr_clf = cefr_data["model"]
# cefr_scaler = cefr_data["scaler"]
# cefr_features = cefr_data["features"]

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# EuroBERT-210M is the default CEFR model: slightly less accurate than
# BritLLM-3B (weighted F1 0.78 vs 0.83 in the CEFR-Cymraeg paper,
# https://aclanthology.org/2026.lrec-1.279/) but ~12x less compute and
# ~1 GB of RAM instead of ~5 GB, so it runs in well under a second per
# text on a small CPU-only server.
# Set PROFFILIADUR_CEFR_MODEL (and optionally PROFFILIADUR_CEFR_REVISION)
# to use another model, e.g. cardiffnlp/WelshCEFR-BritLLM - note that
# needs ~10 GB of RAM in float32. Any model must use the same 4-label
# order (A1, A2, B1, B2) that classify() maps below.
DEFAULT_MODEL_NAME = "cardiffnlp/WelshCEFR-EuroBert"
# Pinned so a push to the Hub repo can't silently change predictions.
DEFAULT_MODEL_REVISION = "56a734b7a11e139bcf21ff0ddb4636da4aa19c46"

MODEL_NAME = os.getenv("PROFFILIADUR_CEFR_MODEL") or DEFAULT_MODEL_NAME
MODEL_REVISION = os.getenv("PROFFILIADUR_CEFR_REVISION") or (
    DEFAULT_MODEL_REVISION if MODEL_NAME == DEFAULT_MODEL_NAME else "main"
)

# This model is expected to be pre-provisioned by setup.sh. In production,
# set HF_HUB_OFFLINE=1 (see .env.example) so a server restart never
# depends on network access - from_pretrained() then fails fast with a
# clear error instead of silently downloading if the cache is missing.
#
# No trust_remote_code: transformers has EuroBERT built in, and the
# custom modeling code in the Hub repo targets transformers 4.x and fails
# to load on the 5.x pinned in pyproject.toml.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

device = "cpu"

# float32, not float16: most server CPUs have no native float16 support,
# so half precision falls back to a much slower path (EuroBERT was ~12x
# slower in float16 even on an Apple M1, which does support it).
cefr_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    dtype="float32"
)

tokenizer.pad_token = tokenizer.eos_token
cefr_model.config.pad_token_id = tokenizer.pad_token_id

cefr_model.to(device)
cefr_model.eval()

def classify(feature_dict, text):
  result = {}

  # --- Readability prediction ---
  X_read = np.array([feature_dict.get(col, 0) for col in read_features]).reshape(1, -1)
  X_read = read_scaler.transform(X_read)
  pred_r = read_clf.predict(X_read)[0]
  prob_r = read_clf.predict_proba(X_read)[0][pred_r]
  result["readability_label"] = "clear" if pred_r == 0 else "not_clear"
  result["readability_confidence"] = float(prob_r)

  # --- CEFR prediction ---
  # X_cefr = np.array([feature_dict.get(col, 0) for col in cefr_features]).reshape(1, -1)
  # X_cefr = cefr_scaler.transform(X_cefr)
  # pred_c = cefr_clf.predict(X_cefr)[0]
  # prob_c = cefr_clf.predict_proba(X_cefr)[0][pred_c]
  # cefr_map = {0: "A1", 1: "A2", 2: "B1", 3: "B2"}
  # cefr_label = cefr_map.get(pred_c, str(pred_c))  # fallback to string

  # result["cefr_label"] = cefr_label
  # result["cefr_confidence"] = float(prob_c)

  # =========================
  # CEFR (TRANSFORMER MODEL)
  # =========================

  inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
  )

  inputs = {
        k: v.to(device)
        for k, v in inputs.items()
  }

  with torch.no_grad():
      outputs = cefr_model(**inputs)

  logits = outputs.logits

  probs = torch.softmax(logits, dim=-1)

  pred_c = torch.argmax(probs, dim=-1).item()

  confidence = probs[0][pred_c].item()

  # automatic label mapping from model config doesnt map to CEFR levels, so...
  cefr_map = {0: "A1", 1: "A2", 2: "B1", 3: "B2"}
  cefr_label = cefr_map.get(pred_c, str(pred_c))  # fallback to string

  result["cefr_label"] = cefr_label
  result["cefr_confidence"] = confidence

  return result
