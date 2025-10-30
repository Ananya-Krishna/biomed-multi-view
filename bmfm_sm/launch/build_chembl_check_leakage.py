#!/usr/bin/env python3
"""
Build per-target ChEMBL datasets with QC + leakage checks.

- Pull IC50 activities for given ChEMBL targets
- Use pChEMBL (normalized potency); aggregate median per std_smiles
- Add derived labels: log10(IC50 [M]) & log10(IC50 [nM])
- Backfill assay_type / assay_confidence_score from assay table if missing
- Verify leakage vs MMP canonical sets (exact + Tanimoto≥0.95)
- Write full and leakage-free (_nolkg) CSVs + summary
"""

import os
import re
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# ========= CONFIG =========
OUT_DIR = "chembl_built"  # created under CWD

TARGETS = {
    "CHEMBL217": "dopamine_D2_receptor",
    "CHEMBL205": "carbonic_anhydrase_II",
    "CHEMBL251": "adenosine_A2a_receptor",
    "CHEMBL214": "serotonin_1a_receptor",
    "CHEMBL244": "coagulation_factor_X",
    "CHEMBL240": "hERG",
}

MMP_FILES = [
    "mmp_ac_s_distinct_canonical.csv",
    "mmp_ac_s_neg_distinct_canonical.csv",
]

TANIMOTO_THRESHOLD = 0.95
FP_NBITS = 2048
FP_RADIUS = 2
SIM_CHUNK = 10_000

os.makedirs(OUT_DIR, exist_ok=True)

# ========= Helpers =========
_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def coerce_num(x):
    """Robustly convert pChEMBL values to float. Handles numbers, '6.05', and '["6.05"]' strings."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    s = str(x)
    m = _num_re.search(s)
    return float(m.group(0)) if m else np.nan

def simple_standardize(smiles: str):
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    mol = max(frags, key=lambda m: m.GetNumAtoms())
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

def morgan_fp(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FP_NBITS)

def fetch_target_dataframe(tid: str) -> pd.DataFrame:
    """Fetch IC50→pChEMBL rows, backfill assay fields, coerce numerics, aggregate to medians."""
    act = new_client.activity
    q = act.filter(
        target_chembl_id=tid,
        standard_type="IC50",
        pchembl_value__isnull=False,
        standard_relation="=",
    ).only([
        "molecule_chembl_id",
        "canonical_smiles",
        "pchembl_value",
        "assay_type",
        "assay_confidence_score",
        "assay_chembl_id",
        "standard_units",
    ])
    df = pd.DataFrame(q)

    # ---- Backfill assay fields if missing ----
    need_backfill = (
        ("assay_confidence_score" not in df.columns or df["assay_confidence_score"].isna().all()) or
        ("assay_type" not in df.columns or df["assay_type"].isna().all())
    )
    if need_backfill and "assay_chembl_id" in df.columns:
        assays = new_client.assay
        assay_ids = df["assay_chembl_id"].dropna().astype(str).unique().tolist()
        rows = []
        CHUNK = 1000
        for i in range(0, len(assay_ids), CHUNK):
            chunk = assay_ids[i:i+CHUNK]
            a = assays.filter(assay_chembl_id__in=chunk).only(
                ["assay_chembl_id", "assay_type", "confidence_score"]
            )
            rows.extend(list(a))
        if rows:
            adf = pd.DataFrame(rows)
            if "confidence_score" in adf.columns:
                adf = adf.rename(columns={"confidence_score": "assay_confidence_score"})
            keep = [c for c in ["assay_chembl_id", "assay_type", "assay_confidence_score"] if c in adf.columns]
            adf = adf[keep].drop_duplicates()
            df = df.merge(adf, on="assay_chembl_id", how="left", suffixes=("", "_assay"))
            if "assay_type_assay" in df.columns:
                df["assay_type"] = df["assay_type"].combine_first(df["assay_type_assay"])
                df = df.drop(columns=["assay_type_assay"])
            if "assay_confidence_score_assay" in df.columns:
                df["assay_confidence_score"] = df["assay_confidence_score"].combine_first(
                    df["assay_confidence_score_assay"]
                )
                df = df.drop(columns=["assay_confidence_score_assay"])

    # ---- QC filters (best-effort, only if present) ----
    if "assay_type" in df.columns:
        df = df[df["assay_type"].fillna("") == "B"]
    else:
        print("  [warn] 'assay_type' missing; skipping assay_type=='B' filter")

    if "assay_confidence_score" in df.columns:
        df = df[df["assay_confidence_score"].fillna(0) >= 8]
    else:
        print("  [warn] 'assay_confidence_score' missing; skipping confidence>=8 filter")

    # Require basic columns
    if "canonical_smiles" not in df.columns or "pchembl_value" not in df.columns:
        print("  [warn] required columns missing after fetch; returning empty frame")
        return pd.DataFrame(columns=[
            "std_smiles","pchembl_median","n_measurements","any_units","any_mol_id",
            "log_ic50_m","log_ic50_nM","target_chembl_id","target_name"
        ])

    # Coerce pChEMBL to numeric robustly and drop non-numeric
    df["pchembl_value"] = df["pchembl_value"].apply(coerce_num)
    df = df.dropna(subset=["canonical_smiles", "pchembl_value"])
    if df.empty:
        print("  [warn] no rows after QC filters for", tid)

    # Standardize SMILES
    df["std_smiles"] = df["canonical_smiles"].map(simple_standardize)
    df = df.dropna(subset=["std_smiles"])

    # Aggregate replicate measurements per std_smiles (numeric median)
    agg = (
        df.groupby("std_smiles", as_index=False)
          .agg(
              pchembl_median=("pchembl_value", "median"),
              n_measurements=("pchembl_value", "size"),
              any_units=("standard_units", "first"),
              any_mol_id=("molecule_chembl_id", "first"),
          )
    )

    # Derived labels
    agg["log_ic50_m"]  = -agg["pchembl_median"]      # log10(IC50 [M])
    agg["log_ic50_nM"] = 9 - agg["pchembl_median"]   # log10(IC50 [nM])
    agg["target_chembl_id"] = tid
    agg["target_name"] = TARGETS[tid]
    return agg

def load_mmp_sets():
    """Load canonical SMILES from MMP files and precompute fingerprints."""
    can_set: set[str] = set()
    for path in MMP_FILES:
        if not os.path.exists(path):
            print(f"⚠️ MMP file not found: {path} (skipping)")
            continue
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c in ("can_c1", "can_c2")]
        if not cols:
            cols = [c for c in df.columns if c.lower().startswith("can_")]
        for col in cols:
            can_set.update(df[col].dropna().astype(str).tolist())

    fps = []
    for smi in can_set:
        fp = morgan_fp(smi)
        if fp is not None:
            fps.append(fp)
    return can_set, fps

def flag_leakage(df: pd.DataFrame, mmp_exact: set[str], mmp_fps: list) -> pd.DataFrame:
    df = df.copy()
    if "std_smiles" not in df.columns:
        df["std_smiles"] = ""
    df["leak_exact"] = df["std_smiles"].isin(mmp_exact)

    df["leak_tani"] = False
    if mmp_fps and not df.empty:
        for i, smi in enumerate(df["std_smiles"].tolist()):
            fp = morgan_fp(smi)
            if fp is None:
                continue
            found = False
            for j in range(0, len(mmp_fps), SIM_CHUNK):
                sims = DataStructs.BulkTanimotoSimilarity(fp, mmp_fps[j:j+SIM_CHUNK])
                if sims and max(sims) >= TANIMOTO_THRESHOLD:
                    found = True
                    break
            if found:
                df.iat[i, df.columns.get_loc("leak_tani")] = True

    df["leak_any"] = df["leak_exact"] | df["leak_tani"]
    return df

# ========= Main =========
if __name__ == "__main__":
    mmp_exact, mmp_fps = load_mmp_sets()
    print(f"Loaded {len(mmp_exact)} canonical MMP SMILES for leakage checks.")

    summary = []
    for tid, tname in TARGETS.items():
        print(f"\n→ Building dataset for {tname} ({tid}) …")
        agg = fetch_target_dataframe(tid)
        print(f"   {len(agg)} unique std_smiles after QC & aggregation")

        checked = flag_leakage(agg, mmp_exact, mmp_fps)

        out_full  = os.path.join(OUT_DIR, f"{tname}_pchembl.csv")
        out_clean = os.path.join(OUT_DIR, f"{tname}_pchembl_nolkg.csv")
        checked.to_csv(out_full, index=False)
        checked[~checked["leak_any"]].to_csv(out_clean, index=False)

        stats = {
            "target": tname, "tid": tid,
            "n_total": int(len(checked)),
            "n_exact": int(checked["leak_exact"].sum()) if "leak_exact" in checked else 0,
            "n_tani": int(checked["leak_tani"].sum()) if "leak_tani" in checked else 0,
            "n_any": int(checked["leak_any"].sum()) if "leak_any" in checked else 0,
            "n_no_leak": int((~checked["leak_any"]).sum()) if "leak_any" in checked else int(len(checked)),
        }
        summary.append(stats)
        print(f"   leakage — exact: {stats['n_exact']}, tani≥{TANIMOTO_THRESHOLD}: {stats['n_tani']}, any: {stats['n_any']}")

    pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, "leakage_summary.csv"), index=False)
    print("\n✅ Done. Outputs in:", OUT_DIR)

