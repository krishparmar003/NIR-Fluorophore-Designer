"""
generate.py
------------
Step 4 of the pipeline: a lightweight de-novo molecule generator.

Strategy (CPU-only, laptop-friendly — a practical stand-in for the paper's
Reinvent 4 generator):
  1. Take "seed" molecules from the NIR dataset (real red-emitting fluorophores).
  2. Apply random chemically-sensible MUTATIONS to grow a virtual library:
       - attach common substituents (NMe2, OMe, CN, NO2, F, etc.)
       - these EDG/EWG groups are exactly what red-shifts fluorophores
         (matches the donor-acceptor logic discussed in the paper).
  3. Keep only NEW, valid, drug-like molecules (not already in the dataset).
  4. SCORE every candidate with the trained prediction models.
  5. Return the best candidates ranked by a user objective
     (e.g. maximize emission wavelength, or hit a target wavelength).

This needs the trained models in models/ (run train_model.py first).
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, BRICS
from rdkit import RDLogger

from featurize import featurize_pair

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "models")
DATA = os.path.join(ROOT, "data", "nir_dataset.csv")

# Substituents that commonly tune fluorophore wavelength (SMILES fragments).
SUBSTITUENTS = [
    "N(C)C",      # dimethylamino (strong donor -> red shift)
    "OC",         # methoxy (donor)
    "O",          # hydroxy
    "C#N",        # nitrile (acceptor)
    "[N+](=O)[O-]",  # nitro (strong acceptor)
    "F", "Cl", "Br",
    "C", "CC",    # alkyl
    "c1ccccc1",   # phenyl (extends conjugation)
    "C=Cc1ccccc1" # styryl (extends conjugation -> red shift)
]


def _load_models():
    models = {}
    for col in ["absorption_nm", "emission_nm", "plqy", "epsilon"]:
        path = os.path.join(MODEL_DIR, f"model_{col}.pkl")
        if os.path.exists(path):
            models[col] = joblib.load(path)
    return models


def predict_properties(smiles, solvent="O", models=None):
    """Predict the 4 optical properties for one molecule in a given solvent."""
    if models is None:
        models = _load_models()
    feat = featurize_pair(smiles, solvent)
    if feat is None:
        return None
    X = feat.reshape(1, -1)
    out = {}
    for col, bundle in models.items():
        # PCA pipeline (if present) or raw model
        if "pca" in bundle and bundle["pca"] is not None:
            Xt = bundle["scaler"].transform(X)
            Xt = bundle["pca"].transform(Xt)
        else:
            Xt = X
        val = float(bundle["model"].predict(Xt)[0])
        if bundle.get("log_transformed"):
            val = 10 ** val
        # clip quantum yield to its physical range [0, 1]
        if col == "plqy":
            val = min(max(val, 0.0), 1.0)
        out[col] = val
    return out


def mutate(smiles):
    """Return a mutated SMILES by attaching a random substituent, or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    sub = random.choice(SUBSTITUENTS)
    # Combine parent + substituent and let RDKit make a bond at an aromatic H
    try:
        combo = f"{smiles}.{sub}"
        m = Chem.MolFromSmiles(combo)
        if m is None:
            return None
        # crude join: pick an aromatic carbon on parent and first atom of sub
        ed = Chem.RWMol(m)
        parent_atoms = [a.GetIdx() for a in mol.GetAtoms()
                        if a.GetIsAromatic() and a.GetTotalNumHs() > 0]
        if not parent_atoms:
            return None
        a1 = random.choice(parent_atoms)
        a2 = mol.GetNumAtoms()  # first atom of the appended fragment
        ed.AddBond(a1, a2, Chem.BondType.SINGLE)
        newmol = ed.GetMol()
        Chem.SanitizeMol(newmol)
        return Chem.MolToSmiles(newmol)
    except Exception:
        return None


def is_drug_like(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False
    mw = Descriptors.MolWt(m)
    return 150 <= mw <= 900 and m.GetNumAtoms() <= 70


def generate(
    n_candidates=200,
    solvent="O",
    objective="max_emission",
    target_nm=700,
    seed_scaffold=None,
    random_seed=42,
):
    """
    Generate and rank new NIR fluorophore candidates.

    objective:
        "max_emission"   -> push emission as red as possible
        "target"         -> get emission close to target_nm
        "max_brightness" -> maximize plqy * epsilon
    seed_scaffold: optional scaffold name (e.g. 'BODIPY') to seed only from it.
    """
    random.seed(random_seed)
    models = _load_models()
    df = pd.read_csv(DATA)
    if seed_scaffold:
        df = df[df["scaffold"] == seed_scaffold]
    seeds = df["smiles"].dropna().unique().tolist()
    known = set(seeds)
    random.shuffle(seeds)

    results, tries = [], 0
    max_tries = n_candidates * 40
    while len(results) < n_candidates and tries < max_tries:
        tries += 1
        parent = random.choice(seeds)
        child = mutate(parent)
        if not child or child in known or not is_drug_like(child):
            continue
        known.add(child)
        props = predict_properties(child, solvent, models)
        if props is None:
            continue
        # only keep genuinely NIR-ish candidates
        if props["emission_nm"] < 600:
            continue
        bright = props["plqy"] * props["epsilon"]
        if objective == "max_emission":
            score = props["emission_nm"]
        elif objective == "target":
            score = -abs(props["emission_nm"] - target_nm)
        else:  # max_brightness
            score = bright
        results.append({
            "smiles": child,
            "parent": parent,
            "absorption_nm": round(props["absorption_nm"], 1),
            "emission_nm": round(props["emission_nm"], 1),
            "plqy": round(props["plqy"], 3),
            "epsilon": round(props["epsilon"], 0),
            "brightness": round(bright, 0),
            "score": round(score, 2),
        })

    out = pd.DataFrame(results).sort_values("score", ascending=False)
    return out.reset_index(drop=True)


if __name__ == "__main__":
    print("Generating NIR fluorophore candidates (max emission)...")
    cands = generate(n_candidates=30, objective="max_emission")
    print(cands.head(15).to_string(index=False))
