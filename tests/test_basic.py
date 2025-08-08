from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT      = Path(__file__).resolve().parent.parent          # repo root
ARTIFACTS = ROOT / "artifacts"

# ---------- helpers ---------------------------------------------------------

def load_model():
    """Load the calibrated model saved in artifacts/"""
    return joblib.load(ARTIFACTS / "calibrated_model.joblib")

def _feature_names(model):
    """Return the list of feature names, whether the model is wrapped or not."""
    if hasattr(model, "feature_name_"):
        return model.feature_name_
    if hasattr(model, "estimator") and hasattr(model.estimator, "feature_name_"):
        return model.estimator.feature_name_
    raise AttributeError("Could not find feature_name_ on model or its estimator")

def make_dummy_frame(model, n_rows=4):
    """Create an all-zeros DF with the right columns for quick smoke tests."""
    cols = _feature_names(model)
    return pd.DataFrame(np.zeros((n_rows, len(cols))), columns=cols)
# ---------- tests -----------------------------------------------------------

def test_model_file_exists():
    """Model artefact is present and non-empty."""
    model_path = ARTIFACTS / "calibrated_model.joblib"
    assert model_path.exists() and model_path.stat().st_size > 10_000  # ~KB+

def test_model_predict_proba_bounds():
    """predict_proba returns values in [0,1] and shape matches input rows."""
    model = load_model()
    X = make_dummy_frame(model, n_rows=7)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape[0] == 7
    assert np.all((proba >= 0) & (proba <= 1))

def test_threshold_in_config():
    """Chosen threshold lives in config.json and is reasonable (0-0.5)."""
    cfg = json.loads((ARTIFACTS / "config.json").read_text())
    thr = cfg["threshold"]
    assert 0.0 < thr < 0.5, f"Suspicious threshold: {thr}"
