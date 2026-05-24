"""
extract_data.py
-----------------
Step 1 of the pipeline.

Takes the FULL FluoDB dataset (data/FluoDB_full.csv) and produces a clean,
FILTERED dataset containing only NIR / red-emitting fluorophores
(emission wavelength > 600 nm).

This is the core requirement: we do NOT use the whole dataset, only the
long-wavelength (>600 nm) part, which is the most useful region for deep-tissue
bioimaging.

Run:
    python src/extract_data.py
Output:
    data/nir_dataset.csv          -> cleaned, filtered dataset used for modelling
    data/nir_dataset_report.txt   -> human-readable summary of what was kept
"""

import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit parsing warnings

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
EMISSION_CUTOFF_NM = 600.0          # keep ONLY emission > 600 nm
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_PATH = os.path.join(ROOT, "data", "FluoDB_full.csv")
OUT_PATH = os.path.join(ROOT, "data", "nir_dataset.csv")
REPORT_PATH = os.path.join(ROOT, "data", "nir_dataset_report.txt")

# Columns in the raw FluoDB.csv
COL_ABS = "absorption/nm"
COL_EM = "emission/nm"
COL_PLQY = "plqy"
COL_EPS = "e/m-1cm-1"
COL_SMILES = "smiles"
COL_SOLVENT = "solvent"
COL_SCAFFOLD = "tag_name"


def canonical_smiles(smi):
    """Return canonical SMILES if valid, else None."""
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def main():
    print(f"[1/5] Loading raw dataset: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"      raw rows: {len(df)}")

    # ------------------------------------------------------------------
    # Filter 1: emission > 600 nm   (THE main filter for this project)
    # ------------------------------------------------------------------
    print(f"[2/5] Filtering emission > {EMISSION_CUTOFF_NM:.0f} nm ...")
    df = df[df[COL_EM].notna()]
    df = df[df[COL_EM] > EMISSION_CUTOFF_NM].copy()
    print(f"      rows after emission filter: {len(df)}")

    # ------------------------------------------------------------------
    # Filter 2: keep only chemically valid SMILES (and canonicalise)
    # ------------------------------------------------------------------
    print("[3/5] Validating & canonicalising SMILES with RDKit ...")
    df["smiles_canonical"] = df[COL_SMILES].apply(canonical_smiles)
    df = df[df["smiles_canonical"].notna()].copy()
    print(f"      rows after SMILES validation: {len(df)}")

    # ------------------------------------------------------------------
    # Tidy up columns & rename to friendly names
    # ------------------------------------------------------------------
    keep = {
        "smiles_canonical": "smiles",
        COL_SOLVENT: "solvent",
        COL_ABS: "absorption_nm",
        COL_EM: "emission_nm",
        COL_PLQY: "plqy",
        COL_EPS: "epsilon",
        COL_SCAFFOLD: "scaffold",
    }
    df = df[list(keep.keys())].rename(columns=keep)

    # Drop exact duplicate (molecule, solvent) rows, keep first
    before = len(df)
    df = df.drop_duplicates(subset=["smiles", "solvent"]).reset_index(drop=True)
    print(f"[4/5] Removed {before - len(df)} duplicate molecule/solvent rows.")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    df.to_csv(OUT_PATH, index=False)
    print(f"[5/5] Saved cleaned NIR dataset -> {OUT_PATH}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    lines = []
    lines.append("NIR fluorophore dataset (emission > 600 nm)")
    lines.append("=" * 50)
    lines.append(f"Total rows (molecule-solvent pairs): {len(df)}")
    lines.append(f"Unique molecules: {df['smiles'].nunique()}")
    lines.append("")
    lines.append("Non-null counts per property:")
    for c in ["absorption_nm", "emission_nm", "plqy", "epsilon"]:
        lines.append(f"  {c:>14}: {df[c].notna().sum()}")
    lines.append("")
    lines.append("Emission (nm) stats:")
    em = df["emission_nm"]
    lines.append(f"  min={em.min():.0f}  max={em.max():.0f}  mean={em.mean():.0f}")
    lines.append("")
    lines.append("Top scaffolds:")
    for name, cnt in df["scaffold"].value_counts().head(15).items():
        lines.append(f"  {str(name):>16}: {cnt}")
    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
