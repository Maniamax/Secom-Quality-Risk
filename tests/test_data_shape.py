import pandas as pd
from pathlib import Path

DATA = Path("data") / "secom_data.csv"        # adjust filename
LABEL = "label"

def test_data_shape():
    df = pd.read_csv(DATA)
    assert LABEL in df.columns, "label column missing"
    assert df[LABEL].notna().all(), "label has NaNs"
    n_pos = df[LABEL].sum()
    n_neg = len(df) - n_pos
    assert n_pos > 0 and n_neg > 0, "dataset must have both classes"
