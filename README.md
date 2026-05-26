# 🔬 NIR Fluorophore Designer

> **🚀 Live App:** [nir-fluorophore-designer.streamlit.app](https://nir-fluorophore-designer.streamlit.app/)

Predict optical properties and design **new near-infrared (NIR) fluorophores** —
molecules that **emit above 600 nm**, the most useful region for deep-tissue
bioimaging.

This project is an educational reimplementation inspired by the **FLAME** paper
(*A modular artificial intelligence framework to facilitate fluorophore design*,
Zhu et al., *Nature Communications* 2025). Instead of using the whole FluoDB
database, it focuses **only on the long-wavelength (emission > 600 nm) subset**.

---

## ✨ Features

- **Predict** — paste any molecule (SMILES) + pick a solvent, get its four optical
  properties instantly.
- **Generate** — create brand-new NIR fluorophore candidates, ranked by an
  objective (push emission as red as possible / hit a target wavelength /
  maximize brightness).
- **Deployed** — runs fully in the browser, no setup needed:
  👉 **[Try it live](https://nir-fluorophore-designer.streamlit.app/)**

---

## 🧪 What it does (pipeline)

| Step | File | Output |
|------|------|--------|
| 1. Extract NIR data (emission > 600 nm) from FluoDB | `src/extract_data.py` | `data/nir_dataset.csv` |
| 2. Train 4 property-prediction models | `src/train_model.py` | `models/*.pkl`, `metrics.json` |
| 3. Generate & score new candidates | `src/generate.py` | ranked candidate list |
| 4. Web app (predict + generate) | `app.py` | Streamlit UI |

The four predicted optical properties:
`absorption_nm`, `emission_nm`, `plqy` (quantum yield), `epsilon` (molar absorption).

---

## ⚡ Quick start (run locally)

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. run the pipeline (optional — trained models are already included)
python src/extract_data.py      # builds data/nir_dataset.csv
python src/train_model.py       # trains models into models/

# 3. launch the app
streamlit run app.py
```

Or open **`NIR_Fluorophore_Designer.ipynb`** and run the cells top-to-bottom to
see every step (extraction → analysis → model → generator) with outputs.

---

## 📊 Model performance (NIR subset, test set)

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

## 🤖 GNN vs Gradient Boosting (model comparison)

The paper uses a **graph neural network (GNN/MPNN)**. To justify the model
choice here, both approaches were trained on the same NIR subset:

| Model | Emission R² | Notes |
|-------|:-----------:|-------|
| Graph Neural Network (MPNN) | 0.61 | learns from raw molecular graph |
| **Gradient Boosting (LightGBM)** | **0.78** | uses pre-computed chemistry fingerprints |

**Finding:** On the full FluoDB the GNN wins, but on this smaller NIR subset
(~4.7k rows) gradient boosting outperforms the GNN. GNNs are data-hungry and
learn representations from scratch, whereas gradient boosting leverages
pre-computed fingerprints, making it more sample-efficient on limited data.
For a focused, deployable tool, gradient boosting is the better practical choice.

> See `NIR_AllProperties_GNN_vs_GB_Colab.ipynb` for the full comparison across
> all four properties (best run on Google Colab with a GPU).

---

## 🚀 Deployment

This app is deployed on **Streamlit Community Cloud**:
👉 **[nir-fluorophore-designer.streamlit.app](https://nir-fluorophore-designer.streamlit.app/)**

To deploy your own copy:
1. Push the repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo.
3. Set the main file to `app.py`.

Deploy-support files included: `requirements.txt`, `packages.txt` (system libs
for RDKit drawing), and `runtime.txt` (pins the Python version). The trained
`models/*.pkl` are committed, so the app works immediately without retraining.

---

## 📁 Project structure

```
nir-fluorophore-designer/
├── app.py                              # Streamlit web app
├── requirements.txt / packages.txt / runtime.txt
├── NIR_Fluorophore_Designer.ipynb      # full pipeline notebook
├── data/
│   ├── FluoDB_full.csv                 # original full database
│   └── nir_dataset.csv                 # filtered emission > 600 nm dataset
├── models/                             # 4 trained models + metrics.json
└── src/
    ├── extract_data.py                 # emission > 600 nm filter + cleaning
    ├── featurize.py                    # SMILES → features
    ├── train_model.py                  # trains the 4 models
    └── generate.py                     # molecule generator
```

---

## 📝 License

**Copyright © 2026 Krish Parmar. All Rights Reserved.**

This project is released under a **proprietary / All-Rights-Reserved license**
(see the [`LICENSE`](LICENSE) file). The original source code, trained models,
app, and documentation may not be used, copied, modified, distributed, hosted,
or deployed without the **prior written permission** of the copyright holder.

**Third-party data notice:** the project uses the **FluoDB** database from the
FLAME project, which is distributed under **CC BY-NC-ND 4.0**. Accordingly, the
dataset and any results derived from it are for **non-commercial use only**, and
the original dataset is credited as below. The CC BY-NC-ND 4.0 terms apply to
the data and take precedence wherever the data is concerned.

For permission requests, contact: **krishparmar003@gmail.com**

---

## 🙏 Acknowledgements & citation

Data source — FluoDB from the FLAME project:

> Zhu, Y. et al. *A modular artificial intelligence framework to facilitate
> fluorophore design.* **Nature Communications** 16, 3598 (2025).
> Repository: https://github.com/ChemloverYuchen/FLAME

This is an independent educational reimplementation for research/learning
purposes.

---

## 💡 Why this matters

NIR fluorophores (emission > 600 nm) are valuable because near-infrared light
penetrates deeper into tissue, produces less background noise, and is gentler on
cells — making them ideal for deep-tissue bioimaging. This tool aims to make
designing such molecules faster and more accessible, right from the browser.

---

<sub>© 2026 Krish Parmar. All Rights Reserved. Built with RDKit, LightGBM, and Streamlit.</sub>
