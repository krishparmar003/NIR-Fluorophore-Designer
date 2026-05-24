# 🔬 NIR Fluorophore Designer

Predict optical properties and design **new near-infrared (NIR) fluorophores**
— molecules that **emit above 600 nm**, the most useful region for deep-tissue
bioimaging.

This project is an educational reimplementation inspired by the **FLAME** paper
(*A modular artificial intelligence framework to facilitate fluorophore design*,
Zhu et al., *Nature Communications* 2025). Instead of using the whole FluoDB
database, we focus **only on the long-wavelength (emission > 600 nm) subset**.

---

## What it does

| Step | File | Output |
|------|------|--------|
| 1. Extract NIR data (emission > 600 nm) from FluoDB | `src/extract_data.py` | `data/nir_dataset.csv` |
| 2. Train 4 property-prediction models | `src/train_model.py` | `models/*.pkl`, `metrics.json` |
| 3. Generate & score new candidates | `src/generate.py` | ranked candidate list |
| 4. Web app (predict + generate) | `app.py` | Streamlit UI |

The four predicted optical properties:
`absorption_nm`, `emission_nm`, `plqy` (quantum yield), `epsilon` (molar absorption).

---

## Quick start (local)

```bash
# 1. create environment & install
pip install -r requirements.txt

# 2. run the full pipeline
python src/extract_data.py      # builds data/nir_dataset.csv
python src/train_model.py       # trains models into models/

# 3. launch the app
streamlit run app.py
```

Or just open **`NIR_Fluorophore_Designer.ipynb`** and run the cells top-to-bottom
to see every step (extraction → analysis → model → generator) with outputs.

---

## Model performance (NIR subset, test set)

| Property | MAE | R² |
|----------|-----|-----|
| Absorption (nm) | ~14 nm | 0.92 |
| Emission (nm) | ~15 nm | 0.79 |
| Quantum yield ΦPL | ~0.10 | 0.73 |
| Molar abs. ε (log₁₀) | ~0.14 | 0.80 |

> Emission R² looks lower than absorption mostly because the >600 nm filter
> shrinks emission's spread (std ≈ 53 nm vs 92 nm for absorption); the MAE is
> actually similar for both. Emission is also an excited-state property and
> carries the noisy Stokes shift, which is inherently harder to predict.

---

## Deploy (free)

**Streamlit Community Cloud** (easiest):
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo.
3. Set the main file to `app.py`. Done.

The trained `models/*.pkl` are committed, so the app works immediately without
retraining on the server.

---

## Notes

- **Why LightGBM, not the paper's graph neural network (FLSF)?**
  FLSF needs a GPU. The fingerprint + gradient-boosting approach here runs in
  seconds on a laptop, deploys to free CPU tiers, and keeps comparable accuracy
  on the NIR subset.
- **Data source:** FluoDB from the FLAME repo
  (`github.com/ChemloverYuchen/FLAME`), CC BY-NC-ND. Used for education/research.
