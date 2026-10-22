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

#MODEL_NAME = "cardiffnlp/WelshCEFR-BritLLM"
MODEL_NAME = "cardiffnlp/WelshCEFR-EuroBert"

# This model is expected to be pre-provisioned by setup.sh. In production,
# set HF_HUB_OFFLINE=1 (see .env.example) so a server restart never
# depends on network access - from_pretrained() then fails fast with a
# clear error instead of silently downloading if the cache is missing.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device = "cpu"

cefr_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    dtype="float16"
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
