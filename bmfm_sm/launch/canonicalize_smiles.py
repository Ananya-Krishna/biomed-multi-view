#!/usr/bin/env python3
import pandas as pd
from rdkit import Chem

def canonicalize_smiles(smi: str) -> str:
    """
    Return a canonical, isomeric SMILES or None if RDKit fails.
    """
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # keep only the largest fragment (drops salts)
    frags = Chem.GetMolFrags(mol, asMols=True)
    mol   = max(frags, key=lambda m: m.GetNumAtoms())
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def process_file(input_csv: str):
    df = pd.read_csv(input_csv)
    # Canonicalize the true SMILES columns c1 and c2
    df["can_c1"] = df["c1"].map(canonicalize_smiles)
    df["can_c2"] = df["c2"].map(canonicalize_smiles)

    before = len(df)
    df = df.dropna(subset=["can_c1", "can_c2"])
    after = len(df)
    print(f"{input_csv}: dropped {before - after} rows with unparsable SMILES, {after} remain")

    output_csv = input_csv.replace(".csv", "_canonical.csv")
    df.to_csv(output_csv, index=False)
    print(f"→ Wrote {output_csv}")

if __name__ == "__main__":
    for fname in ["mmp_ac_s_distinct.csv", "mmp_ac_s_neg_distinct.csv"]:
        process_file(fname)

