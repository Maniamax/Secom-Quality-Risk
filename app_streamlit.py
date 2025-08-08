import json
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path

# ---------- Load artifacts ----------
ART = Path("artifacts")
cal = joblib.load(ART/"calibrated_model.joblib")
lgbm_full = joblib.load(ART/"lgbm_full.joblib")

with open(ART/"config.json") as f:
    cfg = json.load(f)

thr_default = float(cfg["threshold"])
C_FN, C_FP = cfg["costs"]["FN"], cfg["costs"]["FP"]
watched = cfg["watched_features"]
baseline = pd.read_csv(ART/"drift_baseline.csv").set_index("feature")

# ---------- Helpers ----------
def expected_cost(y_hat, thr, c_fn=C_FN, c_fp=C_FP, y_true=None):
    pred = (y_hat >= thr).astype(int)
    fp = int(pred.sum())  # without labels we treat all flagged as workload; for expected cost we need y_true
    if y_true is None:
        return {"workload": pred.mean(), "tp": np.nan, "fp": fp, "fn": np.nan, "tn": np.nan, "cost": c_fp*fp}
    tp = int(((pred==1) & (y_true==1)).sum())
    fn = int(((pred==0) & (y_true==1)).sum())
    tn = int(((pred==0) & (y_true==0)).sum())
    cost = c_fn*fn + c_fp*fp
    return {"workload": pred.mean(), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "cost": cost}

def ewma(x, lam=0.2):
    z = []
    prev = x.iloc[0]
    for v in x:
        prev = lam*v + (1-lam)*prev
        z.append(prev)
    return pd.Series(z, index=x.index)

def ewma_limits(sd, lam=0.2, k=3.0):
    return k * np.sqrt(lam/(2-lam)) * sd

def drift_flags(df):
    # simple per-feature EWMA vs fixed limits; expect rows in time order
    alerts = {}
    for f in watched:
        if f not in df.columns: 
            alerts[f] = {"frac_alarm": np.nan}
            continue
        mu, sd = baseline.loc[f, ["mu","sd"]]
        z = ewma(df[f].fillna(mu))
        lim = ewma_limits(sd)
        alarm = (z > mu+lim) | (z < mu-lim)
        # throttle: need 2 in a row
        alarm2 = alarm & alarm.shift(1, fill_value=False)
        alerts[f] = {"frac_alarm": float(alarm2.mean())}
    # require ≥2 features alarming at a row to count a final alert
    A = pd.DataFrame({f: pd.Series(dtype=bool) for f in watched})
    for f in watched:
        if f in df.columns:
            mu, sd = baseline.loc[f, ["mu","sd"]]
            z = ewma(df[f].fillna(mu))
            lim = ewma_limits(sd)
            alarm = (z > mu+lim) | (z < mu-lim)
            A[f] = (alarm & alarm.shift(1, fill_value=False)).reindex(df.index, fill_value=False)
    final_alert = (A.sum(axis=1) >= 2)
    return alerts, float(final_alert.mean())

# ---------- UI ----------
st.title("Quality Risk & Drift Monitor")

st.sidebar.header("Settings")
thr = st.sidebar.slider("Decision threshold (calibrated)", min_value=0.0, max_value=0.5, value=float(thr_default), step=0.005)
c_fn = st.sidebar.number_input("Cost of FN (missed bad batch)", value=float(C_FN), min_value=0.0)
c_fp = st.sidebar.number_input("Cost of FP (extra review)", value=float(C_FP), min_value=0.0)

st.write("Upload a CSV with the same columns as training (feature names must match).")
file = st.file_uploader("CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)
    missing_cols = [c for c in lgbm_full.feature_name_ if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in lgbm_full.feature_name_]

    if missing_cols:
        st.error(f"Missing {len(missing_cols)} expected columns (showing first 10): {missing_cols[:10]}")
        st.stop()

    X = df[lgbm_full.feature_name_].copy()
    proba = cal.predict_proba(X)[:,1]

    # headline metrics (no labels case)
    res = expected_cost(proba, thr, c_fn, c_fp, y_true=df[df.columns[-1]] if "label" in df.columns else None)

    st.subheader("Headline")
    col1, col2, col3 = st.columns(3)
    col1.metric("Flagged workload", f"{res['workload']*100:.1f}%")
    col2.metric("Threshold", f"{thr:.3f}")
    col3.metric("Expected cost", f"{res['cost']:.1f}")

    # top risky rows
    st.subheader("Top risk rows")
    topn = min(20, len(df))
    idx = np.argsort(-proba)[:topn]
    tbl = pd.DataFrame({"row_idx": idx, "risk_prob": proba[idx]})
    st.dataframe(tbl, use_container_width=True)

    # drift
    st.subheader("Drift check (EWMA on watched features)")
    alerts, final_rate = drift_flags(X)
    st.write(f"Final drift alert rate (≥2 features, 2 consecutive): **{final_rate*100:.1f}%**")
    drift_tbl = pd.DataFrame(alerts).T.sort_values("frac_alarm", ascending=False)
    st.dataframe(drift_tbl, use_container_width=True)

    # extras
    if extra_cols:
        st.info(f"Ignored {len(extra_cols)} extra columns not seen in training (first 10): {extra_cols[:10]}")
else:
    st.info("Waiting for a CSV…")
