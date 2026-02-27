#!/usr/bin/env python3
"""
run_molnet_predictive_custom.py

Single entry-point for MoleculeNet property prediction on:
  - BACE (binary classification)
  - Lipophilicity (regression)

Supports two feature families:
  - bmfm: on-the-fly BMFM embeddings (orig) or projected embeddings (proj_triplet / proj_ntxent)
  - fingerprint: RDKit fingerprints (Morgan/MACCS) -> MLP reg/class head

Designed to run from bmfm_sm/launch.

Inputs: CSV with columns: smiles,label (index column allowed).
Outputs: output_dir/<dataset>/<feature_family>/<rep_or_fpname>/seed_<seed>/...
  - best_ckpt.pth
  - val/test preds csv
  - metrics.json
  - loss_curve.png
  - summary.csv at output root

 Example usage:
 python run_molnet_predictive_custom.py \
  --dataset lipophilicity \
  --csv /mnt/data/lipophilicity.csv \
  --output-dir ./molnet_runs \
  --feature-family bmfm \
  --rep proj_ntxent \
  --proj-ckpt /path/to/head_ntxent.pth \
  --split scaffold --split-seed 42 --seeds 3
"""

import os, json, argparse, random, hashlib
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

# RDKit for fingerprints + scaffold splitting
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Optional UMAP etc not needed here
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# BMFM imports (same style as your IC50 BMFM script)
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel, LateFusionStrategy
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import Graph2dFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.text_finetune_dataset import TextFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.image_finetune_dataset import ImageFinetuneDataPipeline


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Repro ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- Dataset/task config ----------------
def infer_task(dataset: str) -> str:
    d = dataset.lower()
    if d == "bace":
        return "classification"
    if d == "lipophilicity":
        return "regression"
    raise ValueError(f"Unknown dataset: {dataset}. Use bace|lipophilicity")


# ---------------- Splits ----------------
def murcko_scaffold(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc) if sc is not None else None

def scaffold_split_indices(smiles: List[str], train_frac=0.8, val_frac=0.1, seed=42):
    rng = random.Random(seed)
    buckets = {}
    for i, s in enumerate(smiles):
        sc = murcko_scaffold(s) or f"_NOSCAF_{i}"
        buckets.setdefault(sc, []).append(i)
    scafs = list(buckets.keys())
    rng.shuffle(scafs)

    n = len(smiles)
    tr, va, te = set(), set(), set()
    counts = {"tr": 0, "va": 0, "te": 0}

    tr_target = int(train_frac * n)
    va_target = int(val_frac * n)

    for sc in scafs:
        idxs = buckets[sc]
        if counts["tr"] < tr_target:
            tr.update(idxs); counts["tr"] += len(idxs)
        elif counts["va"] < va_target:
            va.update(idxs); counts["va"] += len(idxs)
        else:
            te.update(idxs); counts["te"] += len(idxs)

    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))

def random_split_indices(n: int, train_frac=0.8, val_frac=0.1, seed=42):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    tr_end = int(train_frac * n)
    va_end = int((train_frac + val_frac) * n)
    return idx[:tr_end], idx[tr_end:va_end], idx[va_end:]


# ---------------- Fingerprints ----------------
def fp_morgan(smiles: str, radius: int, nbits: int, use_chirality: bool, use_features: bool):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(
        m, radius, nBits=nbits, useChirality=use_chirality, useFeatures=use_features
    )
    arr = np.zeros((nbits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def fp_maccs(smiles: str):
    from rdkit.Chem import MACCSkeys
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    bv = MACCSkeys.GenMACCSKeys(m)
    arr = np.zeros((bv.GetNumBits(),), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def build_fingerprint_matrix(smiles: List[str], fp_type: str, fp_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    feats = []
    keep = []
    for i, s in enumerate(smiles):
        if fp_type == "morgan":
            v = fp_morgan(
                s,
                radius=int(fp_cfg["radius"]),
                nbits=int(fp_cfg["nbits"]),
                use_chirality=bool(fp_cfg["use_chirality"]),
                use_features=bool(fp_cfg["use_features"]),
            )
        elif fp_type == "maccs":
            v = fp_maccs(s)
        else:
            raise ValueError(f"Unknown fp_type: {fp_type}")

        if v is None:
            continue
        feats.append(v)
        keep.append(i)

    X = np.stack(feats, axis=0).astype(np.float32)
    keep_idx = np.array(keep, dtype=int)
    return X, keep_idx

def canonical_fp_name(fp_type: str, fp_cfg: Dict[str, Any]) -> Tuple[str, str, str]:
    fp_params_json = json.dumps(fp_cfg, sort_keys=True, separators=(",", ":"))
    fp_id = hashlib.md5((fp_type + "|" + fp_params_json).encode("utf-8")).hexdigest()[:8]
    parts = [fp_type]
    for k in ["radius","nbits","use_chirality","use_features"]:
        if k in fp_cfg:
            parts.append(f"{k}={fp_cfg[k]}")
    return " / ".join(parts), fp_id, fp_params_json


# ---------------- BMFM embeddings ----------------
class ProjectionHead(nn.Module):
    # matches your IC50 head format: Linear -> ReLU -> Linear
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    def forward(self, x): return self.net(x)

def load_bmfm_backbone(hf_model: str, ckpt_path: str | None = None):
    model = SmallMoleculeMultiViewModel.from_pretrained(
        model_path=hf_model,
        fusion_strategy=LateFusionStrategy.ATTENTIONAL,
        inference_mode=True,
        huggingface=True,
    ).to(DEVICE).eval()

    # Optional: load a fine-tuned checkpoint (if you have one)
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location=DEVICE)
        # Accept either {"state_dict": ...} or raw state dict
        state_dict = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    return model

def infer_bmfm_dim(model) -> int:
    sm = "CC"
    inp = {}
    inp.update(Graph2dFinetuneDataPipeline.smiles_to_graph_format(sm))
    inp.update(TextFinetuneDataPipeline.smiles_to_text_format(sm))
    inp.update(ImageFinetuneDataPipeline.smiles_to_image_format(sm))
    for k,v in inp.items():
        if torch.is_tensor(v):
            inp[k] = v.to(DEVICE)
    with torch.no_grad():
        out = model(inp)
        vec = out[0] if isinstance(out, tuple) else out
        return int(vec.squeeze().shape[-1])

def load_projection_head(head_ckpt: str, in_dim: int) -> ProjectionHead:
    state = torch.load(head_ckpt, map_location=DEVICE)
    # state dict keys like "0.weight", "2.weight"
    hidden = state["0.weight"].shape[0]
    out_dim = state["2.weight"].shape[0]
    ph = ProjectionHead(in_dim, hidden, out_dim).to(DEVICE)
    ph.net.load_state_dict(state, strict=True)
    ph.eval()
    return ph

def embed_smiles_bmfm(model, smiles: List[str], rep: str, proj_head: ProjectionHead | None) -> Tuple[np.ndarray, np.ndarray]:
    embs = []
    keep = []
    for i, sm in enumerate(smiles):
        inp = {}
        inp.update(Graph2dFinetuneDataPipeline.smiles_to_graph_format(sm))
        inp.update(TextFinetuneDataPipeline.smiles_to_text_format(sm))
        inp.update(ImageFinetuneDataPipeline.smiles_to_image_format(sm))
        for k,v in inp.items():
            if torch.is_tensor(v):
                inp[k] = v.to(DEVICE)
        with torch.no_grad():
            out = model(inp)
            base = (out[0] if isinstance(out, tuple) else out).squeeze()
            if rep == "orig":
                vec = base
            else:
                if proj_head is None:
                    raise ValueError(f"rep={rep} requires --proj-ckpt")
                vec = proj_head(base)
        embs.append(vec.detach().cpu().numpy())
        keep.append(i)
    X = np.vstack(embs).astype(np.float32)
    return X, np.array(keep, dtype=int)


# ---------------- Models (MLP heads) ----------------
class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

class MLPBinaryClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)


# ---------------- Metrics ----------------
def metrics_classification(y_true: np.ndarray, probs: np.ndarray, threshold=0.5) -> Dict[str, Any]:
    y_true = y_true.astype(int)
    probs = probs.astype(float)
    y_pred = (probs >= threshold).astype(int)

    out = {}
    try: out["roc_auc"] = float(roc_auc_score(y_true, probs))
    except Exception: out["roc_auc"] = float("nan")
    try: out["avg_precision"] = float(average_precision_score(y_true, probs))
    except Exception: out["avg_precision"] = float("nan")

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out

def metrics_regression(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    y_true = y_true.astype(float)
    pred = pred.astype(float)
    rmse = math.sqrt(mean_squared_error(y_true, pred))
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
    }


# ---------------- Train loop ----------------
import math

def train_eval(
    task: str,
    X_tr, y_tr, X_va, y_va, X_te, y_te,
    out_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden: int,
    dropout: float,
    patience: int,
):
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    Xtr = torch.tensor(X_tr_s, dtype=torch.float32)
    Xva = torch.tensor(X_va_s, dtype=torch.float32)
    Xte = torch.tensor(X_te_s, dtype=torch.float32)

    ytr = torch.tensor(y_tr.astype(np.float32), dtype=torch.float32)
    yva = torch.tensor(y_va.astype(np.float32), dtype=torch.float32)
    yte = torch.tensor(y_te.astype(np.float32), dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)

    if task == "classification":
        model = MLPBinaryClassifier(X_tr.shape[1], hidden=hidden, dropout=dropout).to(DEVICE)
        pos = float((y_tr == 1).sum())
        neg = float((y_tr == 0).sum())
        pos_w = (neg / max(pos, 1.0)) if pos > 0 else 1.0
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))
    else:
        model = MLPRegressor(X_tr.shape[1], hidden=hidden, dropout=dropout).to(DEVICE)
        loss_fn = nn.MSELoss()

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def eval_loss(loader):
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                pred = model(xb)
                losses.append(loss_fn(pred, yb).item())
        return float(np.mean(losses)) if losses else float("inf")

    best = float("inf")
    best_state = None
    bad = 0
    hist = {"tr_loss": [], "va_loss": []}

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        tr_loss = float(np.mean(tr_losses))
        va_loss = eval_loss(va_loader)
        hist["tr_loss"].append(tr_loss)
        hist["va_loss"].append(va_loss)

        if va_loss < best - 1e-6:
            best = va_loss
            best_state = {
                "model": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "epoch": ep,
                "val_loss": va_loss,
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    torch.save(best_state, out_dir / "best_ckpt.pth")

    plt.figure()
    plt.plot(hist["tr_loss"], label="train")
    plt.plot(hist["va_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title(f"{task} loss (seed={seed})")
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()

    # Reload best
    model.load_state_dict(best_state["model"])
    model.eval()

    def predict(loader):
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(DEVICE)
                out = model(xb).detach().cpu().numpy()
                preds.append(out)
        return np.concatenate(preds, axis=0)

    va_raw = predict(va_loader)
    te_raw = predict(te_loader)

    if task == "classification":
        va_probs = 1 / (1 + np.exp(-va_raw))
        te_probs = 1 / (1 + np.exp(-te_raw))
        va_metrics = metrics_classification(y_va, va_probs)
        te_metrics = metrics_classification(y_te, te_probs)
        pd.DataFrame({"y_true": y_va.astype(int), "p_hat": va_probs}).to_csv(out_dir / "val_preds.csv", index=False)
        pd.DataFrame({"y_true": y_te.astype(int), "p_hat": te_probs}).to_csv(out_dir / "test_preds.csv", index=False)
    else:
        va_metrics = {
            "rmse": float(math.sqrt(mean_squared_error(y_va, va_raw))),
            "mae": float(mean_absolute_error(y_va, va_raw)),
            "r2": float(r2_score(y_va, va_raw)),
        }
        te_metrics = {
            "rmse": float(math.sqrt(mean_squared_error(y_te, te_raw))),
            "mae": float(mean_absolute_error(y_te, te_raw)),
            "r2": float(r2_score(y_te, te_raw)),
        }
        pd.DataFrame({"y_true": y_va.astype(float), "y_hat": va_raw}).to_csv(out_dir / "val_preds.csv", index=False)
        pd.DataFrame({"y_true": y_te.astype(float), "y_hat": te_raw}).to_csv(out_dir / "test_preds.csv", index=False)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"val": va_metrics, "test": te_metrics, "val_loss": best_state["val_loss"]}, f, indent=2)

    return best_state["val_loss"], va_metrics, te_metrics


# ---------------- Main ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["bace", "lipophilicity"])
    p.add_argument("--csv", required=True, help="Path to dataset csv (must have smiles,label columns)")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--feature-family", required=True, choices=["bmfm", "fingerprint"])
    p.add_argument("--rep", default="orig", choices=["orig", "proj_triplet", "proj_ntxent"],
                   help="BMFM representation; proj_* requires --proj-ckpt")
    p.add_argument("--hf-model", default="ibm/biomed.sm.mv-te-84m")
    p.add_argument("--bmfm-ckpt", default=None, help="Optional fine-tuned BMFM checkpoint to load into backbone")
    p.add_argument("--proj-ckpt", default=None, help="Projection head .pth (Linear-ReLU-Linear state dict)")

    # fingerprint config
    p.add_argument("--fp-type", default="morgan", choices=["morgan", "maccs"])
    p.add_argument("--fp-radius", type=int, default=2)
    p.add_argument("--fp-nbits", type=int, default=2048)
    p.add_argument("--fp-use-chirality", action="store_true")
    p.add_argument("--fp-use-features", action="store_true")

    # split
    p.add_argument("--split", default="scaffold", choices=["scaffold", "random"])
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)

    # training
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)

    return p.parse_args()


def main():
    args = parse_args()
    task = infer_task(args.dataset)

    out_root = Path(args.output_dir) / args.dataset / args.feature_family
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "smiles" not in df.columns or "label" not in df.columns:
        raise ValueError(f"CSV must contain columns smiles,label. Found: {df.columns.tolist()}")

    smiles_all = df["smiles"].astype(str).tolist()
    y_all = df["label"].values

    # split first (on smiles), then build features for each split with consistent filtering
    if args.split == "scaffold":
        tr_idx, va_idx, te_idx = scaffold_split_indices(
            smiles_all, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.split_seed
        )
    else:
        tr_idx, va_idx, te_idx = random_split_indices(
            len(smiles_all), train_frac=args.train_frac, val_frac=args.val_frac, seed=args.split_seed
        )

    def subset(idx):
        return [smiles_all[i] for i in idx], y_all[idx]

    smiles_tr, y_tr = subset(tr_idx)
    smiles_va, y_va = subset(va_idx)
    smiles_te, y_te = subset(te_idx)

    # enforce task typing
    if task == "classification":
        y_tr = y_tr.astype(int)
        y_va = y_va.astype(int)
        y_te = y_te.astype(int)
        print("Class balance:", float(y_tr.mean()), float(y_va.mean()), float(y_te.mean()))
    else:
        y_tr = y_tr.astype(float)
        y_va = y_va.astype(float)
        y_te = y_te.astype(float)

    # build features
    run_tag = ""
    meta = {"dataset": args.dataset, "task": task, "split": args.split, "split_seed": args.split_seed}

    if args.feature_family == "fingerprint":
        fp_cfg = {
            "radius": args.fp_radius,
            "nbits": args.fp_nbits,
            "use_chirality": bool(args.fp_use_chirality),
            "use_features": bool(args.fp_use_features),
        } if args.fp_type == "morgan" else {}

        fp_name, fp_id, fp_params_json = canonical_fp_name(args.fp_type, fp_cfg)
        run_tag = f"{fp_name}__{fp_id}".replace(" ", "")
        meta.update({"fp_type": args.fp_type, "fp_name": fp_name, "fp_id": fp_id, "fp_params_json": fp_params_json})

        X_tr, keep_tr = build_fingerprint_matrix(smiles_tr, args.fp_type, fp_cfg)
        X_va, keep_va = build_fingerprint_matrix(smiles_va, args.fp_type, fp_cfg)
        X_te, keep_te = build_fingerprint_matrix(smiles_te, args.fp_type, fp_cfg)

        y_tr = y_tr[keep_tr]; y_va = y_va[keep_va]; y_te = y_te[keep_te]

    else:
        # BMFM embeddings on-the-fly
        run_tag = args.rep
        meta.update({"rep": args.rep, "hf_model": args.hf_model, "bmfm_ckpt": args.bmfm_ckpt, "proj_ckpt": args.proj_ckpt})

        model = load_bmfm_backbone(args.hf_model, ckpt_path=args.bmfm_ckpt)
        in_dim = infer_bmfm_dim(model)

        proj_head = None
        if args.rep != "orig":
            if not args.proj_ckpt:
                raise ValueError(f"--rep {args.rep} requires --proj-ckpt")
            proj_head = load_projection_head(args.proj_ckpt, in_dim)

        X_tr, keep_tr = embed_smiles_bmfm(model, smiles_tr, args.rep, proj_head)
        X_va, keep_va = embed_smiles_bmfm(model, smiles_va, args.rep, proj_head)
        X_te, keep_te = embed_smiles_bmfm(model, smiles_te, args.rep, proj_head)

        y_tr = y_tr[keep_tr]; y_va = y_va[keep_va]; y_te = y_te[keep_te]

    # run multiple seeds
    rows = []
    for s in range(args.seeds):
        seed = args.base_seed + s
        out_dir = out_root / run_tag / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "run_meta.json", "w") as f:
            json.dump(meta | {"seed": seed}, f, indent=2)

        val_loss, val_metrics, test_metrics = train_eval(
            task=task,
            X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, X_te=X_te, y_te=y_te,
            out_dir=out_dir,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden=args.hidden,
            dropout=args.dropout,
            patience=args.patience,
        )

        row = {
            "dataset": args.dataset,
            "task": task,
            "feature_family": args.feature_family,
            "run_tag": run_tag,
            "seed": seed,
            "split": args.split,
            "split_seed": args.split_seed,
            "val_loss": float(val_loss),
            **{f"val_{k}": v for k,v in val_metrics.items()},
            **{f"test_{k}": v for k,v in test_metrics.items()},
        }
        # include fp identifiers if present
        for k in ["fp_type","fp_name","fp_id"]:
            if k in meta:
                row[k] = meta[k]
        if "rep" in meta:
            row["rep"] = meta["rep"]
        rows.append(row)

    summary_path = Path(args.output_dir) / "summary.csv"
    df_sum = pd.DataFrame(rows)
    if summary_path.exists():
        df_prev = pd.read_csv(summary_path)
        df_sum = pd.concat([df_prev, df_sum], ignore_index=True)
    df_sum.to_csv(summary_path, index=False)

    print("✅ Done. Wrote:", summary_path)


if __name__ == "__main__":
    main()
