# Readme:

Per‑target CSVs in Chembl Built folder (examples):

```
adenosine_A2a_receptor_pchembl.csv
adenosine_A2a_receptor_pchembl_nolkg.csv
carbonic_anhydrase_II_pchembl.csv
carbonic_anhydrase_II_pchembl_nolkg.csv
coagulation_factor_X_pchembl.csv
coagulation_factor_X_pchembl_nolkg.csv
hERG_pchembl.csv  hERG_pchembl_nolkg.csv
serotonin_1a_receptor_pchembl.csv  serotonin_1a_receptor_pchembl_nolkg.csv
leakage_summary.csv
```

Three main scripts:

- `chembl_by_target.py` — pull/standardize activities per ChEMBL target; write raw per‑target IC50 tables.
- `canonicalize_smiles.py` — canonicalize SMILES in your **reference MMP sets** used for leakage checking.
- `build_chembl_check_leakage.py` — aggregate to median per `std_smiles`, compute pChEMBL & log(IC50), join assay metadata, leakage‑check vs. canonical MMP lists, and emit `_pchembl.csv` and `_pchembl_nolkg.csv` plus a `leakage_summary.csv`.

---

## Quick start (ChEMBL‑only)

```bash
# 1) Canonicalize the MMP reference sets used for leakage checks (edit script if your filenames differ)
python canonicalize_smiles.py

# 2) Build or refresh per‑target ChEMBL datasets and leakage‑free subsets
python build_chembl_check_leakage.py
```

> Swap the CSV path to any other target (e.g., `serotonin_1a_receptor_pchembl_nolkg.csv`).

---

## Script‑by‑script details

### `chembl_by_target.py`

- Targets included (hard‑coded): Dopamine D2 (`CHEMBL217`), Carbonic anhydrase II (`CHEMBL205`), Adenosine A2a (`CHEMBL251`), Serotonin 1a (`CHEMBL214`), Coagulation factor X (`CHEMBL244`), hERG (`CHEMBL240`).
- Queries ChEMBL **activities** with `standard_type='IC50'`.
- Drops rows with null SMILES, standardizes to **`std_smiles`** (largest fragment; sanitized; canonical isomeric SMILES).
- Coerces numeric values and computes **`log_ic50 = log10(standard_value)`**.
- Deduplicates by `std_smiles` and writes `chembl_<target>_ic50.csv` in the current directory.

**How to run**

```bash
python chembl_by_target.py
```

**Inputs**: internet access to ChEMBL; RDKit installed.\
**Outputs**: `chembl_<target>_ic50.csv` with at least `std_smiles`, `standard_value/units/type`, `log_ic50`.

---

### `canonicalize_smiles.py`

- Cleans and canonicalizes SMILES in  **MMP reference files** used for leakage detection (defaults: `mmp_ac_s_distinct.csv`, `mmp_ac_s_neg_distinct.csv`).
- RDKit parse → keep largest fragment → canonical isomeric SMILES.
- Writes `*_canonical.csv` with columns like `can_c1`, `can_c2` and reports dropped unparsable rows.

**How to run**

```bash
python canonicalize_smiles.py
```

**Inputs**: the two MMP CSVs listed at the bottom of the script (edit to change).\
**Outputs**: `mmp_ac_s_distinct_canonical.csv`, `mmp_ac_s_neg_distinct_canonical.csv`.

---

### `build_chembl_check_leakage.py`

-
  - `OUT_DIR = "chembl_built"` (created under the CWD)
  - `TARGETS = {CHEMBL ID → short name}` (the six targets above)
  - `MMP_FILES = ["mmp_ac_s_distinct_canonical.csv", "mmp_ac_s_neg_distinct_canonical.csv"]`
  - Fingerprints: **Morgan** (radius **2**, **2048** bits); Tanimoto threshold **0.95**; chunked similarity for speed.
- Pipeline:
  1. Fetch activities and **pChEMBL** values for each CHEMBL target.
  2. Backfill **assay\_type** and **assay\_confidence\_score** from assay table.
  3. **QC filters**: keep `assay_type=='B'` (binding) and `assay_confidence_score≥8` when available.
  4. Standardize SMILES → aggregate **median per \*\*\*\*\*\*\*\*****`std_smiles`**.
  5. Compute derived columns: `log_ic50_M`, `log_ic50_nM`.
  6. **Leakage checks** against canonical MMP sets:
     - **Exact match** of `std_smiles` in the canonical list.
     - **Near‑duplicate** if Tanimoto≥0.95 vs. MMP fingerprints.
     - Flags: `leak_exact`, `leak_tani`, `leak_any`.
  7. Write outputs per target to `chembl_built/`:
     - `<target>_pchembl.csv` (full, QC’d) and `<target>_pchembl_nolkg.csv` (leak‑free subset).
  8. Emit `chembl_built/leakage_summary.csv` with counts per target.

**How to run**

```bash
python build_chembl_check_leakage.py
```

**Inputs**: canonical MMP CSVs from the previous step; internet access to ChEMBL.\
**Outputs**: files listed above inside `chembl_built/`.

---

## IC50 training scripts (used for ChEMBL‑only training)

1. **`ic50_training_v4.py`** — BMFM‑embedding **regression** (pIC50). Uses the pretrained multi‑view backbone + per‑target projection heads, then fits a small MLP regressor. Supports **ensembling across seeds** with a **fixed split** shared across seeds.
2. **`ic50_training_fingerprints.py`** — **regression** using classical fingerprints (ECFP/MACCS) instead of BMFM embeddings.
3. **`ic50_training_fingerprints_classification.py`** — **classification** on fingerprints (active/inactive from a pIC50 threshold).

### 1) `ic50_training_v4.py` (BMFM embeddings → MLP regressor)

**Data**: per‑target CSVs from `chembl_built/` (prefer the `_pchembl_nolkg.csv` files). The script auto‑accepts **plain** or **`chembl_`****\*‑prefixed** filenames.

**Inputs & defaults (argparse)**

- `--emb-dir` **(req.)**: directory with projection heads: `emb_dir/target_<tid>/head_triplet.pth` and `head_ntxent.pth`.
- `--chembl-dir` **(req.)**: path to `chembl_built/` with the CSVs.
- `--output-dir` **(req.)**: where results go.
- `--hf-model` *(default **************`ibm/biomed.sm.mv-te-84m`**************)*: HF model to load as the BMFM backbone (late‑fusion attentional).
- `--epochs` *(20)*, `--batch-size` *(64)*, `--lr` *(1e-3)*, `--weight-decay` *(1e-3)*, `--hidden` *(256)*, `--patience` *(8)*.
- `--split` *(scaffold|random, default scaffold)* and `--split-seed` *(42)*: **one fixed split** used for all seeds → safe ensembling.
- `--seeds` *(3)*: number of independent trainings on the **same** split.
- `--reps` *("orig,proj\_triplet,proj\_ntxent")*: which representation to use from the backbone: base embedding or through the triplet/NTXent projection heads.

**What it does**

- Loads BMFM backbone; computes **base\_dim** by forwarding a dummy SMILES.
- Loads per‑target projection heads from `--emb-dir`.
- Embeds each SMILES into one of: `orig`, `proj_triplet`, `proj_ntxent`.
- Creates a **shared split** across seeds (Murcko scaffold by default) and for each seed trains an **MLP regressor** on standardized features/labels.
- Logs **per‑seed metrics** (RMSE, MAE, R², Pearson/Spearman, acc@±0.3/±0.5) and performs an **ensemble** (mean of per‑seed predictions) on the fixed test set.

**Outputs** (under `--output-dir` → `target_<tid>/<rep>/`)

- `seed_metrics.csv` (one row per seed), `test_preds_ensemble.csv` (y\_true + ensemble preds), `loss_history_seed{n}.csv`.
- Plots: `loss_seed{n}.png`, `scatter_seed{n}.png`, `scatter_ensemble.png`.
- Top‑level `summary.csv` with ensemble metrics per target × representation.

**Example (local)**

```bash
python ic50_training_v4.py \
  --emb-dir /path/to/mmp_finetune_output \
  --chembl-dir $PWD/chembl_built \
  --output-dir $PWD/ic50_pipeline_results \
  --reps orig,proj_triplet,proj_ntxent \
  --split scaffold --split-seed 42 --seeds 3 \
  --epochs 20 --batch-size 64 --hidden 256
```

**SLURM**: see `ic50_training.sbatch` for a ready‑to‑run example that sets the cluster python path and calls the script with the correct directories.

---

### 2) `ic50_training_fingerprints.py` (fingerprints → MLP regressor)

**Purpose**: drop BMFM embeddings and use **RDKit fingerprints** as features.

**Key args**

- `--chembl-dir`, `--output-dir`, `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--hidden`, `--patience`, `--split`, `--split-seed`, `--seeds` (same semantics as above).
- `--fps` *(default **************`ecfp2048_r2,maccs`**************)*: comma‑separated fingerprint specs.
  - `ecfp{bits}_r{radius}` (e.g., `ecfp2048_r2`, `ecfp1024_r3`); alias `eccp…` accepted.
  - `maccs` (167‑bit MACCS keys).

**What it does**

- Generates specified fingerprints for each SMILES (`std_smiles`), stacks them (if multiple), and standardizes features/labels.
- Uses the **same fixed split across seeds** and trains an MLP regressor per seed; ensembles on the fixed test set.

**Outputs**

- Same artifact pattern as `v4` (per‑seed CSVs/plots + `summary.csv`).

**Example (local)**

```bash
python ic50_training_fingerprints.py \
  --chembl-dir $PWD/chembl_built \
  --output-dir $PWD/ic50_fp_results \
  --fps ecfp2048_r2,maccs --split scaffold --split-seed 42 --seeds 3
```

**SLURM**: see `ic50_training_fingerprints.sbatch`.

---

### 3) `ic50_training_fingerprints_classification.py` (fingerprints → classifier)

**Purpose**: binary classification using fingerprints. Labels are derived from pIC50‑like values with a configurable threshold.

**Key args**

- Same as the regression FP script **plus**:
  - `--threshold` *(6.0)*: pIC50 cutoff for **active=1** (≈1 µM).
  - `--cmp` *(ge|gt|le|lt, default ge)*: comparison to threshold.
  - `--prob-threshold` *(0.5)*: decision threshold on sigmoid outputs when reporting accuracy/PR.

**Training details**

- BCEWithLogitsLoss with `pos_weight` computed from the **training split** to handle class imbalance.
- Per‑seed metrics: **ROC‑AUC**, **PR‑AUC (Average Precision)**, **Accuracy** (at `--prob-threshold`), plus loss curves.
- Ensemble: average seed probabilities on the fixed test set, then compute the same metrics.

**Example (local)**

```bash
python ic50_training_fingerprints_classification.py \
  --chembl-dir $PWD/chembl_built \
  --output-dir $PWD/ic50_cls_results \
  --fps ecfp2048_r2,maccs --split scaffold --split-seed 42 --seeds 3 \
  --threshold 6.0 --cmp ge --prob-threshold 0.5
```

**SLURM**: see `ic50_training_fp_cls.sbatch`.

---

## Data expectations for all IC50 scripts

- **Inputs**: CSVs in `--chembl-dir` with at least `std_smiles` and one of `pchembl_median` or `standard_value` (treated as pIC50‑like). Use the `_pchembl_nolkg.csv` versions to avoid leakage.
- **Splits**: A single **Murcko scaffold** or **random** split is created with `--split-seed` and **reused across all seeds** to make test‑time ensembling valid.
- **Metrics**: regression → RMSE/MAE/R²/Pearson/Spearman/acc@±δ; classification → ROC‑AUC/PR‑AUC/Accuracy.

---

## SBATCH helpers

- `ic50_training.sbatch` → calls `ic50_training_v4.py` with BMFM embeddings on a GPU partition.
- `ic50_training_fingerprints.sbatch` → calls the FP regression script.
- `ic50_training_fp_cls.sbatch` → calls the FP classification script.

Customize the paths in each file to your cluster’s project directories; keep `OMP_NUM_THREADS`/`MKL_NUM_THREADS` tied to `--cpus-per-task` as shown.

