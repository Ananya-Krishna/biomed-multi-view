import os, math, argparse, random
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, matthews_corrcoef,
    confusion_matrix
)

# RDKit only used for scaffold split (matches your fingerprint classification script)
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# BMFM backbone + pipelines (matches ic50_training_v4.py)
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel, LateFusionStrategy
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import Graph2dFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.text_finetune_dataset import TextFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.image_finetune_dataset import ImageFinetuneDataPipeline


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_SEED = 42

# Same six leakage-checked CSVs used in ic50_training_v4.py
TARGET_FILES = {
    "72":  ("dopamine_D2_receptor_pchembl_nolkg.csv",   "Dopamine D2 receptor"),
    "15":  ("carbonic_anhydrase_II_pchembl_nolkg.csv",  "Carbonic anhydrase II"),
    "252": ("adenosine_A2a_receptor_pchembl_nolkg.csv", "Adenosine A2a receptor"),
    "51":  ("serotonin_1a_receptor_pchembl_nolkg.csv",  "Serotonin 1a receptor"),
    "194": ("coagulation_factor_X_pchembl_nolkg.csv",   "Coagulation factor X"),
    "165": ("hERG_pchembl_nolkg.csv",                   "HERG"),
}


# ---------------- Repro ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- Splits (matches fingerprint classification script) ----------------
def murcko_scaffold(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc) if sc is not None else None

def scaffold_split_indices(smiles, train_frac=0.7, val_frac=0.15, seed=BASE_SEED):
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

    def assign(sc_idxs, tgt):
        if tgt == "tr":
            tr.update(sc_idxs); counts["tr"] += len(sc_idxs)
        elif tgt == "va":
            va.update(sc_idxs); counts["va"] += len(sc_idxs)
        else:
            te.update(sc_idxs); counts["te"] += len(sc_idxs)

    tr_target = int(train_frac * n)
    va_target = int(val_frac * n)

    for sc in scafs:
        idxs = buckets[sc]
        if counts["tr"] < tr_target:
            assign(idxs, "tr")
        elif counts["va"] < va_target:
            assign(idxs, "va")
        else:
            assign(idxs, "te")

    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))

def random_split_indices(n, train_frac=0.7, val_frac=0.15, seed=BASE_SEED):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    tr_end = int(train_frac * n)
    va_end = int((train_frac + val_frac) * n)
    return idx[:tr_end], idx[tr_end:va_end], idx[va_end:]


# ---------------- BMFM embedding (matches ic50_training_v4.py) ----------------
def load_backbone(hf_model: str):
    model = SmallMoleculeMultiViewModel.from_pretrained(
        model_path=hf_model,
        fusion_strategy=LateFusionStrategy.ATTENTIONAL,
        inference_mode=True,
        huggingface=True,
    ).to(DEVICE).eval()
    return model

def get_base_dim(backbone) -> int:
    sample = "CC"
    inp = {}
    inp.update(Graph2dFinetuneDataPipeline.smiles_to_graph_format(sample))
    inp.update(TextFinetuneDataPipeline.smiles_to_text_format(sample))
    inp.update(ImageFinetuneDataPipeline.smiles_to_image_format(sample))
    for k, v in inp.items():
        if torch.is_tensor(v):
            inp[k] = v.to(DEVICE)
    out = backbone(inp)
    vec = out[0] if isinstance(out, tuple) else out
    return int(vec.squeeze().shape[-1])

class ProjectionHead(nn.Module):
    # matches ic50_training_v4.py state dict layout: Linear -> ReLU -> Linear
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

def load_heads_for_target(tid: str, in_dim: int, emb_dir: str):
    base = os.path.join(emb_dir, f"target_{tid}")
    heads = {}
    for ht in ("triplet", "ntxent"):
        ckpt = os.path.join(base, f"head_{ht}.pth")
        state = torch.load(ckpt, map_location=DEVICE)
        hidden = state["0.weight"].shape[0]
        proj = state["2.weight"].shape[0]
        ph = ProjectionHead(in_dim, hidden, proj).to(DEVICE)
        ph.net.load_state_dict(state, strict=True)
        ph.eval()
        heads[ht] = ph
    return heads

def embed_smiles(backbone, heads, smiles, rep_type: str) -> np.ndarray:
    """
    rep_type:
      - "orig"
      - "proj_triplet"
      - "proj_ntxent"
    """
    embs = []
    for sm in smiles:
        inp = {}
        inp.update(Graph2dFinetuneDataPipeline.smiles_to_graph_format(sm))
        inp.update(TextFinetuneDataPipeline.smiles_to_text_format(sm))
        inp.update(ImageFinetuneDataPipeline.smiles_to_image_format(sm))
        for k, v in inp.items():
            if torch.is_tensor(v):
                inp[k] = v.to(DEVICE)

        with torch.no_grad():
            out = backbone(inp)
            base_vec = (out[0] if isinstance(out, tuple) else out).squeeze()
            if rep_type == "orig":
                vec = base_vec
            else:
                # proj_triplet -> "triplet", proj_ntxent -> "ntxent"
                key = rep_type.split("_", 1)[1]
                vec = heads[key](base_vec)
            embs.append(vec.detach().cpu().numpy())

    return np.vstack(embs).astype(np.float32)


# ---------------- Classifier ----------------
class MLPBinaryClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def _metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5):
    y_true = y_true.astype(int)
    probs = probs.astype(float)
    y_pred = (probs >= threshold).astype(int)

    out = {}
    # Some targets/splits can be single-class; guard metrics
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, probs))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["avg_precision"] = float(average_precision_score(y_true, probs))
    except Exception:
        out["avg_precision"] = float("nan")

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["tn"] = int(tn); out["fp"] = int(fp); out["fn"] = int(fn); out["tp"] = int(tp)
    return out


def train_one_seed(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    *,
    seed: int,
    out_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden: int,
    dropout: float,
    patience: int,
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Scale embeddings (fit on train only)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    # tensors
    Xtr = torch.tensor(X_tr_s, dtype=torch.float32)
    ytr = torch.tensor(y_tr.astype(np.float32), dtype=torch.float32)
    Xva = torch.tensor(X_va_s, dtype=torch.float32)
    yva = torch.tensor(y_va.astype(np.float32), dtype=torch.float32)
    Xte = torch.tensor(X_te_s, dtype=torch.float32)
    yte = torch.tensor(y_te.astype(np.float32), dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True, drop_last=False)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=batch_size, shuffle=False, drop_last=False)
    te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False, drop_last=False)

    model = MLPBinaryClassifier(in_dim=X_tr.shape[1], hidden=hidden, dropout=dropout).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # class imbalance handling via pos_weight (train split only)
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    if pos == 0:
        pos_w = 1.0
    else:
        pos_w = neg / max(pos, 1.0)
    pos_weight = torch.tensor([pos_w], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float("inf")
    best_state = None
    bad = 0

    hist = {"tr_loss": [], "va_loss": []}

    def eval_loss(loader):
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("inf")
        va_loss = eval_loss(va_loader)

        hist["tr_loss"].append(tr_loss)
        hist["va_loss"].append(va_loss)

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {
                "model": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "pos_weight": pos_w,
                "epoch": ep,
                "val_loss": va_loss,
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    # Save best checkpoint
    ckpt_path = os.path.join(out_dir, "best_ckpt.pth")
    torch.save(best_state, ckpt_path)

    # Plot loss curves
    plt.figure()
    plt.plot(hist["tr_loss"], label="train")
    plt.plot(hist["va_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.title(f"Loss curves (seed={seed})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # Reload best for metrics/preds
    model.load_state_dict(best_state["model"])
    model.eval()

    def predict_probs(loader):
        probs = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(probs, axis=0) if probs else np.array([])

    va_probs = predict_probs(va_loader)
    te_probs = predict_probs(te_loader)

    va_metrics = _metrics_from_probs(y_va, va_probs)
    te_metrics = _metrics_from_probs(y_te, te_probs)

    # Save predictions
    pd.DataFrame({"y_true": y_va.astype(int), "p_hat": va_probs}).to_csv(
        os.path.join(out_dir, "val_preds.csv"), index=False
    )
    pd.DataFrame({"y_true": y_te.astype(int), "p_hat": te_probs}).to_csv(
        os.path.join(out_dir, "test_preds.csv"), index=False
    )

    return {
        "seed": seed,
        "ckpt_path": ckpt_path,
        "val_loss": float(best_state["val_loss"]),
        "val_metrics": va_metrics,
        "test_metrics": te_metrics,
        "test_probs": te_probs,
    }


# ---------------- Main driver ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", required=True, help="Contains target_{tid}/head_triplet.pth + head_ntxent.pth if using proj_* reps")
    p.add_argument("--chembl-dir", required=True, help="Points to chembl_built folder containing the CSVs")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hf-model", default="ibm/biomed.sm.mv-te-84m")

    # classification label
    p.add_argument("--threshold", type=float, default=6.0, help="Binary label: y=1 if pchembl_median >= threshold")

    # training hyperparams
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=8)

    # ensembling & split control (fixed split across seeds is essential)
    p.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    p.add_argument("--split-seed", type=int, default=42, help="one fixed split for all seeds (critical for ensembling)")
    p.add_argument("--seeds", type=int, default=3, help="number of seeds; actual seeds are BASE_SEED..BASE_SEED+seeds-1")

    # representations
    p.add_argument("--reps", default="orig,proj_triplet,proj_ntxent", help="comma-separated: orig,proj_triplet,proj_ntxent")
    return p.parse_args()


def resolve_csv(chembl_dir: str, csv_name: str) -> str:
    path = os.path.join(chembl_dir, csv_name)
    if os.path.exists(path):
        return path
    alt = os.path.join(chembl_dir, "chembl_" + csv_name)
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f"Could not find {csv_name} or chembl_{csv_name} in {chembl_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    reps = [r.strip() for r in args.reps.split(",") if r.strip()]

    print("Config:", vars(args))
    print("DEVICE:", DEVICE)

    # Load backbone once (shared across targets)
    backbone = load_backbone(args.hf_model)
    base_dim = get_base_dim(backbone)
    print("Backbone base embedding dim:", base_dim)

    summary_rows = []

    for tid, (csv_name, tname) in TARGET_FILES.items():
        print(f"\n=== {tname} (target_{tid}) ===")

        csv_path = resolve_csv(args.chembl_dir, csv_name)
        df = pd.read_csv(csv_path)

        # Match ic50_training_v4.py column choices
        smiles = df["std_smiles"].astype(str).tolist()
        y_cont = df["pchembl_median"].astype(float).values
        y_bin = (y_cont >= args.threshold).astype(int)

        # Fixed split (shared across seeds)
        if args.split == "scaffold":
            tr_idx, va_idx, te_idx = scaffold_split_indices(smiles, seed=args.split_seed)
        else:
            tr_idx, va_idx, te_idx = random_split_indices(len(smiles), seed=args.split_seed)

        smiles_tr = [smiles[i] for i in tr_idx]
        smiles_va = [smiles[i] for i in va_idx]
        smiles_te = [smiles[i] for i in te_idx]
        y_tr = y_bin[tr_idx]
        y_va = y_bin[va_idx]
        y_te = y_bin[te_idx]

        print(f"Split sizes: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)} | pos_rate(train)={y_tr.mean():.3f}")

        # Load projection heads for this target only if needed
        need_proj = any(r.startswith("proj_") for r in reps)
        heads = {}
        if need_proj:
            heads = load_heads_for_target(tid, base_dim, args.emb_dir)

        # Precompute embeddings ONCE per split per rep (keeps embedding logic consistent)
        # (If memory is tight, you can stream; but for your typical IC50 CSV sizes this is fine.)
        for rep in reps:
            rep_out = os.path.join(args.output_dir, f"target_{tid}", rep)
            os.makedirs(rep_out, exist_ok=True)

            print(f"  -> Embedding rep={rep}")
            X_tr = embed_smiles(backbone, heads, smiles_tr, rep)
            X_va = embed_smiles(backbone, heads, smiles_va, rep)
            X_te = embed_smiles(backbone, heads, smiles_te, rep)

            # Train multiple seeds with the SAME split
            seed_runs = []
            seed_probs = []
            for s in range(args.seeds):
                seed = BASE_SEED + s
                seed_out = os.path.join(rep_out, f"seed_{seed}")
                print(f"    -> Training seed={seed}")
                run = train_one_seed(
                    X_tr, y_tr, X_va, y_va, X_te, y_te,
                    seed=seed,
                    out_dir=seed_out,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    hidden=args.hidden,
                    dropout=args.dropout,
                    patience=args.patience,
                )
                seed_runs.append(run)
                seed_probs.append(run["test_probs"])

                # per-seed row
                row = {
                    "target": tname,
                    "tid": tid,
                    "rep": rep,
                    "mode": "seed",
                    "seed": seed,
                    "val_loss": run["val_loss"],
                    **{f"val_{k}": v for k, v in run["val_metrics"].items()},
                    **{f"test_{k}": v for k, v in run["test_metrics"].items()},
                }
                summary_rows.append(row)

            # Ensemble over seeds (mean prob)
            probs_mat = np.vstack(seed_probs)  # [S, Ntest]
            ens_probs = probs_mat.mean(axis=0)
            ens_metrics = _metrics_from_probs(y_te, ens_probs)

            # Save ensemble predictions
            pd.DataFrame({"y_true": y_te.astype(int), "p_hat_ensemble": ens_probs}).to_csv(
                os.path.join(rep_out, "test_preds_ensemble.csv"), index=False
            )

            # Add ensemble row
            summary_rows.append({
                "target": tname,
                "tid": tid,
                "rep": rep,
                "mode": "ensemble",
                "seed": -1,
                "val_loss": float("nan"),
                **{f"test_{k}": v for k, v in ens_metrics.items()},
            })

            print(
                f"    Ensemble test: "
                f"AUC={ens_metrics['roc_auc']:.4f} "
                f"AP={ens_metrics['avg_precision']:.4f} "
                f"Acc={ens_metrics['accuracy']:.4f} "
                f"F1={ens_metrics['f1']:.4f}"
            )

    # Write summary
    summary_path = os.path.join(args.output_dir, "summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print("\n✅ Done. Outputs in:", args.output_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
