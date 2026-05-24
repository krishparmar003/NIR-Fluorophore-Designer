"""
featurize.py
-------------
Shared feature-engineering used by BOTH training (train_model.py) and the
Streamlit app. Keeping it in one place guarantees train-time and predict-time
features are identical.

Each (fluorophore, solvent) pair is turned into one numeric vector:
  - Morgan (ECFP) fingerprint of the fluorophore   (FP_BITS bits)
  - a small set of RDKit physico-chemical descriptors of the fluorophore
  - Morgan fingerprint of the solvent              (SOLV_FP_BITS bits)

This is a deliberately lightweight, CPU-only alternative to the paper's
graph neural network (FLSF). It trains in seconds on a laptop and deploys
to free Streamlit/Vercel tiers without a GPU.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

FP_BITS = 1024        # fluorophore fingerprint length
FP_RADIUS = 2
SOLV_FP_BITS = 128    # solvent fingerprint length

# Descriptor functions applied to the fluorophore
_DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt),
    ("LogP", Crippen.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("RingCount", Descriptors.RingCount),
    ("NumHeteroatoms", Descriptors.NumHeteroatoms),
    ("NHOHCount", Descriptors.NHOHCount),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
]

DESCRIPTOR_NAMES = [n for n, _ in _DESCRIPTORS]


def _morgan(mol, n_bits, radius=FP_RADIUS):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def descriptors(mol):
    out = np.zeros((len(_DESCRIPTORS),), dtype=np.float32)
    for i, (_, fn) in enumerate(_DESCRIPTORS):
        try:
            out[i] = fn(mol)
        except Exception:
            out[i] = 0.0
    return out


def featurize_pair(smiles, solvent_smiles):
    """
    Return a single feature vector for (fluorophore, solvent), or None if the
    fluorophore SMILES is invalid. Solvent is optional (defaults to water).
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    fp_fluo = _morgan(mol, FP_BITS)
    desc = descriptors(mol)

    solv_mol = Chem.MolFromSmiles(str(solvent_smiles)) if solvent_smiles else None
    if solv_mol is None:
        solv_mol = Chem.MolFromSmiles("O")  # default = water
    fp_solv = _morgan(solv_mol, SOLV_FP_BITS)

    return np.concatenate([fp_fluo, desc, fp_solv])


def feature_length():
    return FP_BITS + len(_DESCRIPTORS) + SOLV_FP_BITS
