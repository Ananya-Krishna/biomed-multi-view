#!/usr/bin/env python3
# ic50_training_fingerprints_cls.py
# Classification pipeline using RDKit fingerprints (ECFP/MACCS) + MLP.
# - Fixed scaffold/random split shared across seeds → safe ensembling
# - Binary labels from pIC50-like values using --threshold (default 6.0)
# - BCEWithLogitsLoss with class imbalance handling (pos_weight from train)
# - Per-seed + ensemble metrics; train/val loss curves (per-seed + aggregate)

import os, math, argparse, random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, precision_recall_fscore_support, confusion_matrix
)
from scipy.stats import pearsonr, spearmanr  # not used here but harmless if you reuse utilities
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit import DataStructs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_SEED = 42

# Your six leakage-checked CSVs (without the 'chembl_' prefix).
# The script will also try a 'chembl_' prefixed fallback if the plain name isn't found.
TARGET_FILES = {
    "72":  ("dopamine_D2_receptor_pchembl_nolkg.csv",   "Dopamine D2 receptor"),
    "15":  ("carbonic_anhydrase_II_pchembl_nolkg.csv",  "Carbonic anhydrase II"),
    "252": ("adenosine_A2a_receptor_pchembl_nolkg.csv", "Adenosine A2a receptor"),
    "51":  ("serotonin_1a_receptor_pchembl_nolkg.csv",  "Serotonin 1a receptor"),
    "194": ("coagulation_factor_X_pchembl_nolkg.csv",   "Coagulation factor X"),
    "165": ("hERG_pchembl_nolkg.csv",                   "HERG"),
}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def murcko_scaffold(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc) if sc is not None else None

def scaffold_split_indices(smiles, train_frac=0.7, val_frac=0.15, seed=BASE_SEED):
    rng = random.Random(seed)
    buckets = {}
    for i, s in enumerate(smiles):
        sc = murcko_scaffold(s) or f"_NOSCAF_{i}"
        buckets.setdefault(sc, []).append(i)
    scafs = list(buckets.keys()); rng.shuffle(scafs)
    n = len(smiles)
    tr, va, te = set(), set(), set()
    counts = {"tr":0, "va":0, "te":0}
    for sc in scafs:
        tgt = "tr" if counts["tr"]/n < train_frac else ("va" if counts["va"]/n < val_frac else "te")
        idxs = buckets[sc]
        if tgt=="tr": tr.update(idxs); counts["tr"] += len(idxs)
        elif tgt=="va": va.update(idxs); counts["va"] += len(idxs)
        else: te.update(idxs); counts["te"] += len(idxs)
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))

def random_split_indices(n, train_frac=0.7, val_frac=0.15, seed=BASE_SEED):
    idx = np.arange(n); rng = np.random.default_rng(seed); rng.shuffle(idx)
    tr_end, va_end = int(train_frac*n), int((train_frac+val_frac)*n)
    return idx[:tr_end], idx[tr_end:va_end], idx[va_end:]

# ---------------- Fingerprints ----------------
# Spec examples:
#   "ecfp2048_r2" , "ecfp1024_r3" , "maccs" , "eccp2048_r2" (alias for ecfp)
def parse_fp_spec(spec):
    s = spec.lower()
    if s == "maccs":
        return {"type":"maccs"}
    if s.startswith("eccp"):  # alias
        s = "ecfp" + s[len("eccp"):]
    if s.startswith("ecfp"):
        bits = 2048; radius = 2
        rest = s[len("ecfp"):]
        if rest:
            parts = rest.strip("_").split("_")
            for p in parts:
                if p.isdigit(): bits = int(p)
                elif p.startswith("r") and p[1:].isdigit(): radius = int(p[1:])
        return {"type":"ecfp", "nBits": bits, "radius": radius}
    raise ValueError(f"Unrecognized fingerprint spec: {spec}")

def mol_from_smiles(smi):
    return Chem.MolFromSmiles(smi)

def fp_vector_from_mol(mol, fp_cfg):
    if fp_cfg["type"] == "maccs":
        bv = GenMACCSKeys(mol)  # 167 bits
        arr = np.zeros((bv.GetNumBits(),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr
    elif fp_cfg["type"] == "ecfp":
        bv = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=fp_cfg["radius"], nBits=fp_cfg["nBits"]
        )
        arr = np.zeros((bv.GetNumBits(),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bv, arr)
        return arr
    else:
        raise ValueError("Unknown fp type")

def featurize_smiles(smiles, fp_spec):
    fp_cfg = parse_fp_spec(fp_spec)
    X_list, valid_idx = [], []
    for i, sm in enumerate(smiles):
        m = mol_from_smiles(sm)
        if m is None:
            X_list.append(None)
            continue
        X_list.append(fp_vector_from_mol(m, fp_cfg))
        valid_idx.append(i)
    valid_mask = np.array([x is not None for x in X_list], dtype=bool)
    X = np.vstack([x for x in X_list if x is not None]).astype(np.float32)
    return X, valid_mask

# ---- Model ----
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(hidden, max(64, hidden//2)), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(max(64, hidden//2), 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# ---- Metrics helpers ----
def safe_roc_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")

def safe_avg_precision(y_true, y_score):
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")

def cls_metrics(y_true, y_logit, thr=0.5):
    # logits → probs
    y_prob = torch.sigmoid(torch.as_tensor(y_logit)).cpu().numpy()
    y_pred = (y_prob >= thr).astype(np.int32)
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        prec, rec, f1_s, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        precision, recall = float(prec), float(rec)
    except Exception:
        precision = recall = float("nan")
    auroc = safe_roc_auc(y_true, y_prob)
    auprc = safe_avg_precision(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(np.unique(y_true))==2 else (np.nan,)*4
    return {
        "accuracy": acc, "f1": f1, "precision": precision, "recall": recall,
        "auroc": auroc, "auprc": auprc, "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }, y_prob, y_pred

# ---- One-seed training on a FIXED split ----
def train_on_fixed_split(X, y_bin, tr_idx, va_idx, te_idx, args, seed, out_dir):
    set_seed(seed)

    X_tr, X_va, X_te = X[tr_idx], X[va_idx], X[te_idx]
    y_tr, y_va, y_te = y_bin[tr_idx], y_bin[va_idx], y_bin[te_idx]

    # Scale features (bit vectors benefit from centering/variance scaling for MLPs)
    x_scaler = StandardScaler(with_mean=True, with_std=True).fit(X_tr)
    X_tr_s = x_scaler.transform(X_tr); X_va_s = x_scaler.transform(X_va); X_te_s = x_scaler.transform(X_te)

    Ttr = torch.from_numpy(X_tr_s).float().to(DEVICE)
    Tva = torch.from_numpy(X_va_s).float().to(DEVICE)
    Tte = torch.from_numpy(X_te_s).float().to(DEVICE)
    Ytr = torch.from_numpy(y_tr.astype(np.float32)).float().to(DEVICE)
    Yva = torch.from_numpy(y_va.astype(np.float32)).float().to(DEVICE)

    model = MLP(in_dim=X.shape[1], hidden=args.hidden).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Imbalance handling: pos_weight = N_neg / N_pos (computed on TRAIN only)
    n_pos = max(1, int(y_tr.sum()))
    n_neg = max(1, int((1 - y_tr).sum()))
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, verbose=False)

    best_val = float("inf"); best_state = None
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_auroc": [], "val_auprc": []}
    patience = args.patience

    for ep in range(1, args.epochs+1):
        model.train()
        perm = torch.randperm(Ttr.size(0), device=DEVICE)
        # mini-batch loop
        for i in range(0, Ttr.size(0), args.batch_size):
            idx = perm[i:i+args.batch_size]
            pred_logit = model(Ttr[idx])
            loss = loss_fn(pred_logit, Ytr[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Eval on full train/val to log losses & metrics (ES on val loss)
        model.eval()
        with torch.no_grad():
            tr_logit = model(Ttr)
            va_logit = model(Tva)
            tr_loss = float(loss_fn(tr_logit, Ytr).item())
            va_loss = float(loss_fn(va_logit, Yva).item())

            # validation metrics (for monitoring)
            va_metrics, va_prob, _ = cls_metrics(y_va, va_logit.cpu().numpy(), thr=args.prob_threshold)

        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_auroc"].append(va_metrics["auroc"])
        history["val_auprc"].append(va_metrics["auprc"])

        sched.step(va_loss)
        if va_loss < best_val - 1e-6:
            best_val, best_state, patience = va_loss, model.state_dict(), args.patience
        else:
            patience -= 1
            if patience == 0: break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save per-seed training history + plots
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, f"loss_history_seed{seed}.csv"), index=False)

    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train loss")
    plt.plot(history["epoch"], history["val_loss"], label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("BCE loss"); plt.legend(); plt.title("Train/Val loss")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"loss_seed{seed}.png"), dpi=150); plt.close()

    # Test predictions (fixed order across seeds)
    model.eval()
    with torch.no_grad():
        te_logit = model(Tte).cpu().numpy()

    # Metrics on test
    metrics, y_prob, y_pred = cls_metrics(y_te, te_logit, thr=args.prob_threshold)

    # Scatter-style diagnostic: probability histogram split by class
    plt.figure()
    plt.hist(y_prob[y_te==0], bins=30, alpha=0.6, label="inactive (0)")
    plt.hist(y_prob[y_te==1], bins=30, alpha=0.6, label="active (1)")
    plt.xlabel("Predicted probability (active)"); plt.ylabel("Count")
    plt.title("Test predicted probabilities")
    plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"proba_hist_seed{seed}.png"), dpi=150); plt.close()

    # Save test preds
    pd.DataFrame({"y_true": y_te, "y_pred_proba": y_prob, "y_pred_label": y_pred}).to_csv(
        os.path.join(out_dir, f"test_preds_seed{seed}.csv"), index=False
    )
    return metrics, y_te.copy(), te_logit  # return logits for robust ensembling

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chembl-dir", required=True)       # points to chembl_built
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--split", choices=["scaffold","random"], default="scaffold")
    p.add_argument("--split-seed", type=int, default=42, help="one fixed split for all seeds (critical for ensembling)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--fps", default="ecfp2048_r2,maccs",
                   help="Comma-separated fingerprint specs, e.g. ecfp2048_r2,maccs")
    # Classification settings
    p.add_argument("--threshold", type=float, default=6.0,
                   help="pIC50 threshold for 'active' (label=1). Default 6.0 (≈1 µM).")
    p.add_argument("--cmp", choices=["ge","gt","le","lt"], default="ge",
                   help="Comparison for labeling: ge=≥, gt=>, le=≤, lt=<")
    p.add_argument("--prob-threshold", type=float, default=0.5,
                   help="Probability threshold for converting probs→labels (for metrics).")
    return p.parse_args()

def binarize_labels(pIC50_array, thr, cmp):
    if cmp == "ge": return (pIC50_array >= thr).astype(np.int32)
    if cmp == "gt": return (pIC50_array >  thr).astype(np.int32)
    if cmp == "le": return (pIC50_array <= thr).astype(np.int32)
    if cmp == "lt": return (pIC50_array <  thr).astype(np.int32)
    raise ValueError("Unknown comparison op")

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("Config:", vars(args))

    summary = []

    for tid, (csv_name, tname) in TARGET_FILES.items():
        print(f"\n=== {tname} (target_{tid}) ===")
        # Resolve CSV path (with or without 'chembl_' prefix)
        path = os.path.join(args.chembl_dir, csv_name)
        if not os.path.exists(path):
            alt = os.path.join(args.chembl_dir, "chembl_" + csv_name)
            if os.path.exists(alt):
                path = alt
            else:
                raise FileNotFoundError(f"Could not find {csv_name} or chembl_{csv_name} in {args.chembl_dir}")

        df = pd.read_csv(path)
        if "pchembl_median" in df.columns:
            y_all = df["pchembl_median"].astype(float).values
        elif "standard_value" in df.columns:
            # fallback: assumed already pIC50-like per previous pipeline
            y_all = df["standard_value"].astype(float).values
        else:
            raise ValueError("Could not find 'pchembl_median' or 'standard_value' in dataset.")
        smiles_all = df["std_smiles"].astype(str).tolist()

        # Build binary labels once (before masking invalid SMILES, we’ll align later)
        y_bin_all = binarize_labels(y_all, thr=args.threshold, cmp=args.cmp)

        for fp in [r.strip() for r in args.fps.split(",") if r.strip()]:
            print(f"\n-- fingerprint {fp}")
            # Build X and mask invalid SMILES rows
            X_full, valid_mask = featurize_smiles(smiles_all, fp)
            y_full = y_bin_all[valid_mask]
            smiles_valid = [s for s, m in zip(smiles_all, valid_mask) if m]

            # ---- FIXED split computed ONCE per fingerprint ----
            if args.split == "scaffold":
                tr_idx, va_idx, te_idx = scaffold_split_indices(smiles_valid, 0.7, 0.15, seed=args.split_seed)
            else:
                tr_idx, va_idx, te_idx = random_split_indices(len(smiles_valid), 0.7, 0.15, seed=args.split_seed)

            rep_out = os.path.join(args.output_dir, f"target_{tid}", fp)
            os.makedirs(rep_out, exist_ok=True)

            seed_metrics = []
            all_test_logits = []
            y_test_ref = None

            for s in range(args.seeds):
                out_dir = os.path.join(rep_out, f"seed_{s}")
                m, y_te, te_logits = train_on_fixed_split(
                    X_full, y_full, tr_idx, va_idx, te_idx, args, seed=BASE_SEED+s, out_dir=out_dir
                )
                seed_metrics.append({"seed": s, **m})
                all_test_logits.append(te_logits)
                if y_test_ref is None:
                    y_test_ref = y_te

            # --- Aggregate loss curves across seeds (mean ± std) ---
            histories = []
            for s in range(args.seeds):
                hist_path = os.path.join(rep_out, f"seed_{s}", f"loss_history_seed{BASE_SEED+s}.csv")
                # Back-compat: if filename used seed index instead of base+offset
                if not os.path.exists(hist_path):
                    hist_path = os.path.join(rep_out, f"seed_{s}", f"loss_history_seed{BASE_SEED+s}.csv")
                # If still missing, try generic pattern (seed number unknown)
                if not os.path.exists(hist_path):
                    # try to find any loss_history in that folder
                    candidates = [p for p in os.listdir(os.path.join(rep_out, f"seed_{s}")) if p.startswith("loss_history_seed") and p.endswith(".csv")]
                    if candidates:
                        hist_path = os.path.join(rep_out, f"seed_{s}", candidates[0])
                if os.path.exists(hist_path):
                    dfh = pd.read_csv(hist_path)
                    dfh["epoch"] = np.arange(1, len(dfh) + 1)
                    histories.append(dfh)

            if histories:
                max_ep = max(len(h) for h in histories)
                n_seeds = len(histories)
                arr_tr = np.full((n_seeds, max_ep), np.nan, dtype=np.float32)
                arr_va = np.full((n_seeds, max_ep), np.nan, dtype=np.float32)
                for i, h in enumerate(histories):
                    arr_tr[i, :len(h)] = h["train_loss"].values
                    arr_va[i, :len(h)] = h["val_loss"].values
                mean_tr = np.nanmean(arr_tr, axis=0)
                std_tr  = np.nanstd(arr_tr, axis=0)
                mean_va = np.nanmean(arr_va, axis=0)
                std_va  = np.nanstd(arr_va, axis=0)
                ep = np.arange(1, max_ep + 1)

                pd.DataFrame({
                    "epoch": ep,
                    "train_loss_mean": mean_tr, "train_loss_std": std_tr,
                    "val_loss_mean": mean_va,   "val_loss_std": std_va,
                }).to_csv(os.path.join(rep_out, "loss_history_aggregate.csv"), index=False)

                plt.figure()
                plt.plot(ep, mean_tr, label="train loss (mean)")
                plt.fill_between(ep, mean_tr - std_tr, mean_tr + std_tr, alpha=0.2)
                plt.plot(ep, mean_va, label="val loss (mean)")
                plt.fill_between(ep, mean_va - std_va, mean_va + std_va, alpha=0.2)
                plt.xlabel("Epoch")
                plt.ylabel("BCE loss")
                plt.title("Train/Val loss — aggregate across seeds")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(rep_out, "loss_curves_aggregate.png"), dpi=150)
                plt.close()

            # Ensemble: average logits → prob → metrics
            ens_logit = np.mean(np.vstack(all_test_logits), axis=0)
            ens_metrics, ens_prob, ens_pred = cls_metrics(y_test_ref, ens_logit, thr=args.prob_threshold)
            print(f"   Ensemble → Acc {ens_metrics['accuracy']:.3f}  AUROC {ens_metrics['auroc']:.3f}  AUPRC {ens_metrics['auprc']:.3f}  F1 {ens_metrics['f1']:.3f}")

            # Save per-rep artifacts
            pd.DataFrame(seed_metrics).to_csv(os.path.join(rep_out, "seed_metrics.csv"), index=False)
            pd.DataFrame({"y_true": y_test_ref, "y_pred_proba_ensemble": ens_prob, "y_pred_label_ensemble": ens_pred}).to_csv(
                os.path.join(rep_out, "test_preds_ensemble.csv"), index=False
            )

            # Probability histogram for ensemble
            plt.figure()
            plt.hist(ens_prob[y_test_ref==0], bins=30, alpha=0.6, label="inactive (0)")
            plt.hist(ens_prob[y_test_ref==1], bins=30, alpha=0.6, label="active (1)")
            plt.xlabel("Predicted probability (active)"); plt.ylabel("Count")
            plt.title("Test predicted probabilities (ensemble)")
            plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(rep_out, "proba_hist_ensemble.png"), dpi=150); plt.close()

            summary.append({
                "target": tname, "tid": tid, "fp": fp,
                **ens_metrics
            })

    pd.DataFrame(summary).to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    print("\n✅ Done. Outputs in:", args.output_dir)
