"""
app.py  --  NIR Fluorophore Designer (Streamlit)
================================================
Deployable web app. Two tools:

  1. PREDICT  : enter a SMILES + solvent -> get the 4 optical properties.
  2. GENERATE : produce brand-new NIR fluorophore candidates ranked by an
                objective (max emission / target wavelength / max brightness).

The app loads the 4 trained models from models/. If they are missing it
shows instructions to run training first.

Run locally:
    streamlit run app.py
Deploy:
    Push the repo to GitHub -> share.streamlit.io -> point at app.py.
"""

import os
import sys
import pandas as pd
import streamlit as st

# Make src/ importable whether run from repo root or elsewhere
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from rdkit import Chem
from rdkit.Chem import Draw

# ----------------------------------------------------------------------
# Page config & light styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="NIR Fluorophore Designer",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background: #0e1117; }
      .big-metric { font-size: 2rem; font-weight: 700; }
      .subtle { color: #9aa0a6; font-size: 0.85rem; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Model loading (cached)
# ----------------------------------------------------------------------
MODEL_DIR = os.path.join(HERE, "models")
REQUIRED = ["absorption_nm", "emission_nm", "plqy", "epsilon"]

# Common solvents -> SMILES (matches how the dataset stores them)
SOLVENTS = {
    "Water": "O",
    "Methanol": "CO",
    "Ethanol": "CCO",
    "Dichloromethane (DCM)": "ClCCl",
    "Chloroform": "ClC(Cl)Cl",
    "Acetonitrile": "CC#N",
    "DMSO": "CS(C)=O",
    "Toluene": "Cc1ccccc1",
    "THF": "C1CCOC1",
    "DMF": "CN(C)C=O",
}


@st.cache_resource
def models_available():
    return all(
        os.path.exists(os.path.join(MODEL_DIR, f"model_{c}.pkl"))
        for c in REQUIRED
    )


def mol_image(smiles, size=(380, 260)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🔬 NIR Fluorophore Designer")
st.markdown(
    "<span class='subtle'>Predict optical properties and design new "
    "near-infrared (emission &gt; 600 nm) fluorophores. "
    "Built on a filtered subset of the FluoDB database.</span>",
    unsafe_allow_html=True,
)

if not models_available():
    st.error(
        "Trained models not found in `models/`.\n\n"
        "Run the pipeline first:\n"
        "```\npython src/extract_data.py\npython src/train_model.py\n```"
    )
    st.stop()

# import after model check so import errors are clearer
from generate import predict_properties, generate  # noqa: E402

tab_predict, tab_generate, tab_about = st.tabs(
    ["🧪 Predict", "✨ Generate", "ℹ️ About"]
)

# ======================================================================
# TAB 1 -- PREDICT
# ======================================================================
with tab_predict:
    st.subheader("Predict optical properties of a molecule")
    c1, c2 = st.columns([3, 2])
    with c1:
        smiles = st.text_input(
            "Fluorophore SMILES",
            value="CCN(CC)c1ccc2c(c1)oc1cc(=[N+](CC)CC)ccc1c2-c1ccccc1C(=O)O",
            help="Paste any molecule in SMILES notation.",
        )
    with c2:
        solvent_name = st.selectbox("Solvent", list(SOLVENTS.keys()), index=0)

    if st.button("Predict properties", type="primary"):
        props = predict_properties(smiles, SOLVENTS[solvent_name])
        if props is None:
            st.error("Invalid SMILES — could not parse the molecule.")
        else:
            img = mol_image(smiles)
            ic, mc = st.columns([2, 3])
            with ic:
                if img:
                    st.image(img, caption="Parsed structure")
            with mc:
                a, b = st.columns(2)
                a.metric("Absorption λabs", f"{props['absorption_nm']:.0f} nm")
                b.metric("Emission λem", f"{props['emission_nm']:.0f} nm")
                a.metric("Quantum yield ΦPL", f"{props['plqy']:.3f}")
                b.metric("Molar abs. ε", f"{props['epsilon']:,.0f} M⁻¹cm⁻¹")
                stokes = props["emission_nm"] - props["absorption_nm"]
                st.caption(f"Stokes shift ≈ {stokes:.0f} nm  ·  "
                           f"Brightness (ΦPL × ε) ≈ "
                           f"{props['plqy'] * props['epsilon']:,.0f}")
            if props["emission_nm"] > 600:
                st.success("✅ This molecule is predicted to emit in the "
                           "NIR/red region (> 600 nm).")
            else:
                st.info("This molecule is predicted to emit below 600 nm "
                        "(outside the NIR target range).")

# ======================================================================
# TAB 2 -- GENERATE
# ======================================================================
with tab_generate:
    st.subheader("Generate new NIR fluorophore candidates")
    g1, g2, g3 = st.columns(3)
    with g1:
        objective = st.selectbox(
            "Objective",
            ["max_emission", "target", "max_brightness"],
            format_func=lambda x: {
                "max_emission": "Push emission as red as possible",
                "target": "Hit a target emission wavelength",
                "max_brightness": "Maximize brightness (ΦPL × ε)",
            }[x],
        )
    with g2:
        target_nm = st.slider("Target emission (nm)", 600, 1000, 700, 10,
                              disabled=(objective != "target"))
    with g3:
        gen_solvent = st.selectbox("Solvent ", list(SOLVENTS.keys()), index=0,
                                   key="gen_solvent")

    g4, g5 = st.columns(2)
    with g4:
        n_show = st.slider("How many candidates to return", 5, 50, 15, 5)
    with g5:
        scaffold = st.selectbox(
            "Seed scaffold (optional)",
            ["Any", "BODIPY", "Cyanine", "Coumarin", "Porphyrin",
             "Triphenylamine", "Acridines", "Carbazole"],
        )

    if st.button("Generate candidates", type="primary"):
        with st.spinner("Mutating seed molecules and scoring with the model..."):
            df = generate(
                n_candidates=n_show,
                solvent=SOLVENTS[gen_solvent],
                objective=objective,
                target_nm=target_nm,
                seed_scaffold=None if scaffold == "Any" else scaffold,
            )
        if df.empty:
            st.warning("No candidates produced. Try a different scaffold or "
                       "objective.")
        else:
            st.success(f"Generated {len(df)} new candidates.")
            show = df[["smiles", "absorption_nm", "emission_nm",
                       "plqy", "epsilon", "brightness"]]
            st.dataframe(show, use_container_width=True, height=380)

            st.markdown("#### Top candidate")
            top = df.iloc[0]
            tc1, tc2 = st.columns([2, 3])
            with tc1:
                img = mol_image(top["smiles"])
                if img:
                    st.image(img)
            with tc2:
                st.code(top["smiles"], language="text")
                a, b = st.columns(2)
                a.metric("Emission", f"{top['emission_nm']:.0f} nm")
                b.metric("Absorption", f"{top['absorption_nm']:.0f} nm")
                a.metric("ΦPL", f"{top['plqy']:.3f}")
                b.metric("ε", f"{top['epsilon']:,.0f}")

            st.download_button(
                "⬇️ Download candidates as CSV",
                df.to_csv(index=False).encode(),
                file_name="nir_candidates.csv",
                mime="text/csv",
            )

# ======================================================================
# TAB 3 -- ABOUT
# ======================================================================
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
This tool focuses on **near-infrared (NIR) fluorophores** — molecules that
**emit above 600 nm**, the most useful region for deep-tissue bioimaging.

**Pipeline**
1. **Data** — Start from FluoDB (the open-source database from the FLAME paper,
   *Nat. Commun.* 2025) and keep **only the emission > 600 nm** rows.
2. **Features** — Each molecule → Morgan (ECFP) fingerprint + RDKit
   physicochemical descriptors; solvent → its own fingerprint.
3. **Models** — Four gradient-boosted-tree regressors (LightGBM), one per
   optical property. CPU-only, laptop-friendly.
4. **Generator** — Mutates real NIR molecules with wavelength-tuning
   substituents, keeps valid/novel/drug-like ones, and ranks them with the
   trained models.

**Why gradient boosting and not the paper's graph neural network?**
The paper's FLSF model needs a GPU. This fingerprint + LightGBM approach runs
in seconds on a normal laptop and deploys to free hosting tiers, while keeping
comparable accuracy on this NIR subset.
        """
    )
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            m = json.load(f)
        st.markdown("**Test-set performance (emission > 600 nm subset)**")
        rows = []
        for k, v in m["targets"].items():
            rows.append({
                "Property": k, "MAE": v["MAE"], "R²": v["R2"],
                "Unit": v["unit"] or "-",
                "Train/Test": f"{v['n_train']}/{v['n_test']}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption("Data: FluoDB (Zhu et al., Nat. Commun. 2025, CC BY-NC-ND). "
               "This is an educational reimplementation.")
