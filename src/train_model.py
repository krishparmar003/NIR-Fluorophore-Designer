"""
train_model.py
---------------
Step 2 of the pipeline.

Trains FOUR separate regression models (one per optical property) on the
filtered NIR dataset (emission > 600 nm) produced by extract_data.py.

Properties:
    absorption_nm   maximum absorption wavelength
    emission_nm     maximum emission wavelength
    plqy            photoluminescence quantum yield
    epsilon         molar absorption coefficient (log10 transformed internally)

Each model is a Gradient-Boosted Tree regressor. We try LightGBM and fall back
to scikit-learn's HistGradientBoostingRegressor if LightGBM is unavailable, so
the project always runs.

Run:
    python src/train_model.py
Outputs (into models/):
    model_absorption_nm.pkl, model_emission_nm.pkl,
    model_plqy.pkl, model_epsilon.pkl
    metrics.json   -> test-set MAE / R2 for each property
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from featurize import featurize_pair

# Try LightGBM, else fall back to sklearn
try:
    from lightgbm import LGBMRegressor
    def make_regressor():
        return LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            subsample=0.9, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            verbose=-1,
        )
    BACKEND = "LightGBM"
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor
    def make_regressor():
        return HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=63, random_state=42,
        )
    BACKEND = "HistGradientBoosting (sklearn)"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "nir_dataset.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# (column, log10-transform?)
TARGETS = [
    ("absorption_nm", False),
    ("emission_nm", False),
    ("plqy", False),
    ("epsilon", True),     # epsilon spans orders of magnitude -> log10
]


def build_features(df):
    """Return X (features) and a row-mask of valid molecules."""
    feats, mask = [], []
    for smi, solv in zip(df["smiles"], df["solvent"]):
        v = featurize_pair(smi, solv)
        if v is None:
            mask.append(False)
        else:
            mask.append(True)
            feats.append(v)
    return np.array(feats, dtype=np.float32), np.array(mask)


def main():
    print(f"Backend: {BACKEND}")
    print(f"Loading {DATA}")
    df = pd.read_csv(DATA)

    print("Featurizing molecules (this takes ~10-30s)...")
    X_all, mask = build_features(df)
    df = df[mask].reset_index(drop=True)
    print(f"  feature matrix: {X_all.shape}")

    metrics = {"backend": BACKEND, "targets": {}}

    for col, use_log in TARGETS:
        sub = df[df[col].notna()].copy()
        idx = sub.index.values
        X = X_all[idx]
        y = sub[col].values.astype(np.float32)

        if use_log:
            y = np.log10(np.clip(y, 1.0, None))

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = make_regressor()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        mae = float(mean_absolute_error(y_te, pred))
        r2 = float(r2_score(y_te, pred))

        # Report MAE in original units for epsilon (log scale stays log)
        unit = "log10(M-1cm-1)" if use_log else ("nm" if "nm" in col else "")
        metrics["targets"][col] = {
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "n_features": int(X.shape[1]),
            "MAE": round(mae, 4), "R2": round(r2, 4),
            "unit": unit, "log_transformed": use_log,
        }
        print(f"  {col:>14}: MAE={mae:.3f} {unit}  R2={r2:.3f}  "
              f"(train={len(X_tr)}, test={len(X_te)})")

        joblib.dump(
            {"model": model, "log_transformed": use_log},
            os.path.join(MODEL_DIR, f"model_{col}.pkl"),
        )

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved 4 models + metrics.json into {MODEL_DIR}")


if __name__ == "__main__":
    main()
