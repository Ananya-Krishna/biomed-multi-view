#!/usr/bin/env python3
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem

# === 1) CONFIGURATION ===
# Target name → ChEMBL ID mappings
targets = {
    "dopamine_D2_receptor":     "CHEMBL217",  # Dopamine D2 receptor :contentReference[oaicite:0]{index=0}
    "carbonic_anhydrase_II":    "CHEMBL205",  # Carbonic anhydrase II :contentReference[oaicite:1]{index=1}
    "adenosine_A2a_receptor":   "CHEMBL251",  # Adenosine A2a receptor :contentReference[oaicite:2]{index=2}
    "serotonin_1a_receptor":    "CHEMBL214",  # Serotonin 1a (5-HT1a) receptor :contentReference[oaicite:3]{index=3}
    "coagulation_factor_X":     "CHEMBL244",  # Coagulation factor X :contentReference[oaicite:4]{index=4}
    "hERG":                     "CHEMBL240"   # HERG potassium channel :contentReference[oaicite:5]{index=5}
}
STANDARD_TYPE = "IC50"
MAX_RECORDS   = None          # or an int to limit how many per target
OUTPUT_PREFIX = "chembl"

# === 2) SIMPLE SMILES STANDARDIZATION ===
def simple_standardize(smiles):
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # keep only the largest fragment (drops salts)
    frags = Chem.GetMolFrags(mol, asMols=True)
    mol   = max(frags, key=lambda m: m.GetNumAtoms())
    Chem.SanitizeMol(mol)  # kekulize, assign stereochem, etc.
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

# === 3) MAIN LOOP ===
if __name__ == "__main__":
    activity_client = new_client.activity

    for name, target_id in targets.items():
        print(f"\n→ Processing {name} ({target_id})…")

        # 3A) Fetch raw IC50 activities
        query = activity_client.filter(
            target_chembl_id=target_id,
            standard_type=STANDARD_TYPE,
            standard_value__isnull=False
        )
        if MAX_RECORDS:
            query = query[:MAX_RECORDS]

        raw = query.only([
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
            "standard_units"
        ])
        df = pd.DataFrame(raw)
        print(f"  • Pulled {len(df)} raw records")

        # 3B) Drop entries with missing SMILES
        df = df.dropna(subset=["canonical_smiles"])
        print(f"  • {len(df)} records after dropping null SMILES")

        # 3C) Standardize & dedupe
        df["std_smiles"] = df["canonical_smiles"].map(simple_standardize)
        df = df.dropna(subset=["std_smiles"])
        before = len(df)
        df = df.drop_duplicates(subset=["std_smiles"])
        print(f"  • Standardized & deduped: {before} → {len(df)}")

        # 3D) Compute log10(IC50)
        df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
        df = df.dropna(subset=["standard_value"])
        df["log_ic50"] = np.log10(df["standard_value"])

        # 3E) Save to CSV
        out_csv = f"{OUTPUT_PREFIX}_{name}_ic50.csv"
        df.to_csv(out_csv, index=False)
        print(f"  ✓ Wrote {len(df)} records to {out_csv}")

