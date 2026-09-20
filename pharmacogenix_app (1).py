"""
pharmacogenix_app.py
==============================================================================
Streamlit front end for the PharmacoGenix pIC50 predictor.

Loads the artifacts produced by pharmacogenix_pipeline.py:
    results/rf_pIC50_model.pkl
    results/variance_selector.pkl
    results/imputer.pkl
    results/fingerprint_config.json

Run:
    streamlit run pharmacogenix_app.py
==============================================================================
"""
import json
import os

import numpy as np
import streamlit as st
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger

from lipinski_rules import lipinski_rule_of_five

RDLogger.DisableLog('rdApp.*')

ARTIFACT_DIR = "results"


@st.cache_resource
def load_artifacts(artifact_dir):
    model = joblib.load(os.path.join(artifact_dir, 'rf_pIC50_model.pkl'))
    selector = joblib.load(os.path.join(artifact_dir, 'variance_selector.pkl'))
    imputer = joblib.load(os.path.join(artifact_dir, 'imputer.pkl'))
    with open(os.path.join(artifact_dir, 'fingerprint_config.json')) as f:
        fp_config = json.load(f)
    return model, selector, imputer, fp_config


def compute_fingerprint(smiles, radius, n_bits):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.int8).reshape(1, -1), mol


def main():
    st.set_page_config(page_title="PharmacoGenix", page_icon="\U0001F9EA", layout="centered")
    st.title("PharmacoGenix")
    st.caption("Predict pIC50 against MAB erm(41) from a canonical SMILES string.")

    if not os.path.isdir(ARTIFACT_DIR) or not os.path.exists(
            os.path.join(ARTIFACT_DIR, 'rf_pIC50_model.pkl')):
        st.error(
            f"Model artifacts not found in '{ARTIFACT_DIR}/'. Run "
            f"`python pharmacogenix_pipeline.py --data <your_csv>` first to train "
            f"and save the model."
        )
        st.stop()

    model, selector, imputer, fp_config = load_artifacts(ARTIFACT_DIR)

    smiles = st.text_input("Canonical SMILES", placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
    predict_clicked = st.button("Predict pIC50")

    if predict_clicked:
        if not smiles.strip():
            st.warning("Enter a SMILES string first.")
            st.stop()

        fp, mol = compute_fingerprint(smiles.strip(), fp_config['radius'], fp_config['n_bits'])
        if mol is None:
            st.error("RDKit could not parse that SMILES string. Check it and try again.")
            st.stop()

        fp_imputed = imputer.transform(fp)
        fp_selected = selector.transform(fp_imputed)
        pred_pic50 = float(model.predict(fp_selected)[0])

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(Draw.MolToImage(mol, size=(320, 320)), caption="Parsed structure")
        with col2:
            st.metric("Predicted pIC50", f"{pred_pic50:.3f}")
            st.caption("Higher pIC50 = predicted more potent (lower IC50).")

            lip = lipinski_rule_of_five(mol)
            st.write("**Lipinski / drug-likeness**")
            st.write(f"- MW: {lip['MW']:.1f}")
            st.write(f"- LogP: {lip['LogP']:.2f}")
            st.write(f"- H-bond donors: {lip['NumHDonors']}")
            st.write(f"- H-bond acceptors: {lip['NumHAcceptors']}")
            st.write(f"- Lipinski violations: {lip['violations']} "
                     f"({'drug-like' if lip['drug_like'] else 'outside typical drug-like range'})")

        st.info(
            "This prediction reflects the trained Random Forest regressor's estimate only. "
            "It is not a substitute for experimental potency assays."
        )


if __name__ == '__main__':
    main()
