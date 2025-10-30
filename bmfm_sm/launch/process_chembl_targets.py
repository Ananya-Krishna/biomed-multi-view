#!/usr/bin/env python3
import pandas as pd
import os

# === 1) CONFIG: map filenames → ChEMBL target IDs ===
file_map = {
    "chembl_adenosine_A2a_receptor_ic50.csv":   "CHEMBL251",
    "chembl_carbonic_anhydrase_II_ic50.csv":    "CHEMBL205",
    "chembl_coagulation_factor_X_ic50.csv":     "CHEMBL244",
    "chembl_dopamine_D2_receptor_ic50.csv":     "CHEMBL217",
    "chembl_hERG_ic50.csv":                     "CHEMBL240",
    "chembl_serotonin_1a_receptor_ic50.csv":    "CHEMBL214",
}

def process_file(fname, target_id):
    df = pd.read_csv(fname)
    print(f"\n→ {fname}: {len(df)} rows loaded")

    # 1) drop any rows with no standardized SMILES
    if "std_smiles" in df.columns:
        before = len(df)
        df = df.dropna(subset=["std_smiles"])
        print(f"   • dropped {before - len(df)} rows missing std_smiles")

    # 2) dedupe on std_smiles
    before = len(df)
    df = df.drop_duplicates(subset=["std_smiles"])
    print(f"   • deduped: {before} → {len(df)} rows")

    # 3) attach the target ID
    df["target_chembl_id"] = target_id

    # 4) write out
    base, ext = os.path.splitext(fname)
    out = f"{base}_processed{ext}"
    df.to_csv(out, index=False)
    print(f"   ✓ wrote {len(df)} rows to {out}")

if __name__ == "__main__":
    for fname, tid in file_map.items():
        if not os.path.exists(fname):
            print(f"⚠️  file not found: {fname}")
            continue
        process_file(fname, tid)

