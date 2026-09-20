"""
lipinski_rules.py
==============================================================================
Lipinski "Rule of Five" descriptor calculation and drug-likeness evaluation
for the PharmacoGenix pipeline.

Follows the same lightweight, lambda-based accessor style as RDKit's own
rdkit.Chem.Lipinski module (each descriptor is a small named function with a
docstring and a version tag), plus one addition: lipinski_rule_of_five(),
the "custom function ... developed to assess the molecular drug-likeness of
the compounds based on the well-known Lipinski properties" described in the
manuscript methods.

Individual descriptors are computed via RDKit's public rdMolDescriptors /
Descriptors API rather than re-implemented from SMARTS, so this module has
no dependency beyond RDKit itself.

Usage:
    from lipinski_rules import lipinski_descriptors, lipinski_rule_of_five, add_lipinski_columns

    mol = Chem.MolFromSmiles(smiles)
    desc = lipinski_descriptors(mol)          # {'MW': .., 'LogP': .., ...}
    verdict = lipinski_rule_of_five(mol)       # {'violations': 0, 'drug_like': True, ...}
    df = add_lipinski_columns(df, smiles_col='canonical_smiles')
==============================================================================
"""
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def _as_mol(mol_or_smiles):
    """Accepts either an RDKit Mol or a SMILES string; returns a Mol or None."""
    if isinstance(mol_or_smiles, str):
        return Chem.MolFromSmiles(mol_or_smiles)
    return mol_or_smiles


# ------------------------------------------------------------------------
# Individual descriptor accessors (RDKit-backed, one job each)
# ------------------------------------------------------------------------
MolWt = lambda mol: Descriptors.MolWt(mol)
MolWt.__doc__ = "Molecular weight (Daltons)."
MolWt.version = "1.0.0"

MolLogP = lambda mol: Descriptors.MolLogP(mol)
MolLogP.__doc__ = "Crippen-estimated octanol/water partition coefficient (LogP)."
MolLogP.version = "1.0.0"

NumHDonors = lambda mol: rdMolDescriptors.CalcNumHBD(mol)
NumHDonors.__doc__ = "Number of hydrogen bond donors."
NumHDonors.version = "1.0.0"

NumHAcceptors = lambda mol: rdMolDescriptors.CalcNumHBA(mol)
NumHAcceptors.__doc__ = "Number of hydrogen bond acceptors."
NumHAcceptors.version = "1.0.0"

NumRotatableBonds = lambda mol: rdMolDescriptors.CalcNumRotatableBonds(mol)
NumRotatableBonds.__doc__ = "Number of rotatable bonds."
NumRotatableBonds.version = "1.0.0"

TPSA = lambda mol: rdMolDescriptors.CalcTPSA(mol)
TPSA.__doc__ = "Topological polar surface area (\u00c5\u00b2)."
TPSA.version = "1.0.0"

_DESCRIPTOR_FUNCS = {
    'MW': MolWt,
    'LogP': MolLogP,
    'NumHDonors': NumHDonors,
    'NumHAcceptors': NumHAcceptors,
    'NumRotatableBonds': NumRotatableBonds,
    'TPSA': TPSA,
}


def lipinski_descriptors(mol_or_smiles):
    """
    Computes the full set of descriptors used elsewhere in this study
    (MW, LogP, NumHDonors, NumHAcceptors, plus NumRotatableBonds and TPSA
    for completeness). Returns None for an unparseable SMILES.
    """
    mol = _as_mol(mol_or_smiles)
    if mol is None:
        return None
    return {name: fn(mol) for name, fn in _DESCRIPTOR_FUNCS.items()}


# ------------------------------------------------------------------------
# Custom Rule-of-Five evaluator (the manuscript's "custom function")
# ------------------------------------------------------------------------
# Standard thresholds (Lipinski et al., 1997):
#   MW <= 500 Da, LogP <= 5, HBD <= 5, HBA <= 10
# A compound is conventionally still considered drug-like with at most one
# violation of these four rules.
LIPINSKI_THRESHOLDS = {'MW': 500.0, 'LogP': 5.0, 'NumHDonors': 5, 'NumHAcceptors': 10}
MAX_ALLOWED_VIOLATIONS = 1


def lipinski_rule_of_five(mol_or_smiles, thresholds=None, max_violations=MAX_ALLOWED_VIOLATIONS):
    """
    Custom drug-likeness evaluator: computes the four core Lipinski
    descriptors and flags which ones violate the Rule of Five, following
    the manuscript's description of a bespoke Lipinski-evaluation function
    (as distinct from just calling a library's built-in Ro5 check).

    Returns a dict:
        {
          'MW': ..., 'LogP': ..., 'NumHDonors': ..., 'NumHAcceptors': ...,
          'MW_violation': bool, 'LogP_violation': bool,
          'NumHDonors_violation': bool, 'NumHAcceptors_violation': bool,
          'violations': int,       # total rule violations (0-4)
          'drug_like': bool,       # violations <= max_violations
        }
    Returns None if the SMILES/Mol could not be parsed.
    """
    mol = _as_mol(mol_or_smiles)
    if mol is None:
        return None

    thresholds = thresholds or LIPINSKI_THRESHOLDS
    values = {
        'MW': MolWt(mol),
        'LogP': MolLogP(mol),
        'NumHDonors': NumHDonors(mol),
        'NumHAcceptors': NumHAcceptors(mol),
    }

    result = dict(values)
    violations = 0
    for prop, limit in thresholds.items():
        is_violation = values[prop] > limit
        result[f'{prop}_violation'] = is_violation
        violations += int(is_violation)

    result['violations'] = violations
    result['drug_like'] = violations <= max_violations
    return result


# ------------------------------------------------------------------------
# Batch helper for dataframes (mirrors the study's "final dataset ... saved
# as a .csv file" step)
# ------------------------------------------------------------------------
def add_lipinski_columns(df, smiles_col='canonical_smiles', drop_unparseable=False):
    """
    Computes MW, LogP, NumHDonors, NumHAcceptors, NumRotatableBonds, TPSA,
    violation flags, total violation count, and a drug_like boolean for
    every row, appending them as new columns.

    If drop_unparseable=True, rows whose SMILES RDKit cannot parse are
    removed; otherwise their new columns are left as NaN.
    """
    records = []
    for smi in df[smiles_col]:
        verdict = lipinski_rule_of_five(smi)
        records.append(verdict if verdict is not None else {})

    lipinski_df = pd.DataFrame(records, index=df.index)
    out = pd.concat([df, lipinski_df], axis=1)

    if drop_unparseable:
        out = out.dropna(subset=['MW']).reset_index(drop=True)

    return out


if __name__ == '__main__':
    # Quick self-check when run directly: python lipinski_rules.py
    demo_smiles = ["CC(=O)Oc1ccccc1C(=O)O", "Not a real SMILES", "CCO"]
    for smi in demo_smiles:
        print(smi, "->", lipinski_rule_of_five(smi))
