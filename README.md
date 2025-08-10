# Quality Risk Classifier & Drift Monitor

Predicts high-risk semiconductor batches (SECOM dataset) and adds lightweight, explainable **drift monitoring**.  
Model: **LightGBM** with **isotonic calibration**. Decisions use a costed threshold (default **FN:FP = 10:1**). A **Streamlit** app provides a simple UI for upload, triage, and drift status.

**Current results on test set:**  
For sample cost function, cost reduced from 260 to 193.  
ROC-AUC = .737 • PR-AUC = .147.
---
# Key capabilities
- Cost-aware classification: choose a threshold that minimises expected cost for a given cost function.
- Probability calibration: isotonic mapping improves decision quality without changing ranking.
- Explainability: SHAP highlights top drivers.
- Drift monitoring: EWMA control charts on top SHAP features; throttled alerts to reduce noise.
- **Streamlit demo:** upload CSVs, see flagged rows, workload %, expected cost, and drift status.
---
# Dataset
- **SECOM** (Semiconductor Manufacturing) from the UCI ML Repository.  
  https://archive.ics.uci.edu/dataset/179/secom
---
## Project structuregit status
git add README.md
git commit -m "docs: add professional README"
git push
## Project structure
├── app_streamlit.py 
├── artifacts/ 
│ ├── calibrated_model.joblib
│ ├── lgbm_full.joblib
│ ├── config.json 
│ └── drift_baseline.csv
├── notebooks/
│ └── Secom.ipynb
├── src/ 
│ ├── init.py
│ ├── data.py
│ ├── features.py
│ └── train.py
├── tests/a
│ └── test_basic.py
├── environment.yml
├── .github/workflows/ci.yml 
├── .gitignore
└── README.md
## Usage
Train & calibrate 
Open notebooks/Secom.ipynb and run cells to:
	load data,
	fit LightGBM,
	calibrate (isotonic),
	select threshold via cost sweep,
	compute SHAP, and
	write artifacts to artifacts/.
Run the streamlit app "app_streamlit.py"
Upload a CSV with the same feature columns used in training. 
Config: adjust artifacts/config.json to change threshold or cost ratios.
---
## Quick start
	```bash
	conda env create -f environment.yml
	conda activate qrisk
	streamlit run app_streamlit.py
---
