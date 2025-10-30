#!/usr/bin/env python3
# ic50_training_v4_fix.py
# Fixes ensemble crash by enforcing the SAME scaffold/random split across all seeds.
# Also supports either with/without 'chembl_' filename prefix automatically.

import os, math, argparse, random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# BMFM backbone
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel, LateFusionStrategy
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import Graph2dFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.text_finetune_dataset import TextFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.image_finetune_dataset import ImageFinetuneDataPipeline

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

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

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

# ---- BMFM backbone & heads ----
def load_backbone(hf_model):
    model = SmallMoleculeMultiViewModel.from_pretrained(
        model_path=hf_model,
        fusion_strategy=LateFusionStrategy.ATTENTIONAL,
        inference_mode=True,
        huggingface=True
    ).to(DEVICE).eval()
    return model

def get_base_dim(backbone):
    sample = "CC"
    inp = {}
    inp.update(Graph2dFinetuneDataPipeline.smiles_to_graph_format(sample))
    inp.update(TextFinetuneDataPipeline.smiles_to_text_format(sample))
    inp.update(ImageFinetuneDataPipeline.smiles_to_image_format(sample))
    for k,v in inp.items():
        if torch.is_tensor(v): inp[k] = v.to(DEVICE)
    out = backbone(inp)
    vec = out[0] if isinstance(out, tuple) else out
    return int(vec.squeeze().shape[-1])

class ProjectionHead(nn.Sequential):
    def __init__(self, in_dim, hidden, proj):
        super().__init__(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, proj))

def load_heads_for_target(tid, in_dim, emb_dir):
    base = os.path.join(emb_dir, f"target_{tid}")
    heads = {}
    for ht in ("triplet","ntxent"):
        ckpt = os.path.join(base, f"head_{ht}.pth")
        state = torch.load(ckpt, map_location=DEVICE)
        hidden = state["0.weight"].shape[0]
        proj   = state["2.weight"].shape[0]
        ph = ProjectionHead(in_dim, hidden, proj).to(DEVICE)
        ph.load_state_dict(state, strict=True); ph.eval()
        heads[ht] = ph
    return heads

def embed_smiles(backbone, heads, smiles, rep_type):
    embs = []
    for sm in smiles:
        inp = {}
        inp.update(Graph2dFinetuneDataPipeline.smiles_to_graph_format(sm))
        inp.update(TextFinetuneDataPipeline.smiles_to_text_format(sm))
        inp.update(ImageFinetuneDataPipeline.smiles_to_image_format(sm))
        for k,v in inp.items():
            if torch.is_tensor(v): inp[k] = v.to(DEVICE)
        with torch.no_grad():
            out = backbone(inp)
            base_vec = (out[0] if isinstance(out, tuple) else out).squeeze()
            vec = base_vec if rep_type=="orig" else heads[rep_type.split("_")[1]](base_vec)
            embs.append(vec.detach().cpu().numpy())
    return np.vstack(embs).astype(np.float32)

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

# ---- One-seed training on a FIXED split ----
def train_on_fixed_split(X, y, tr_idx, va_idx, te_idx, args, seed, out_dir):
    set_seed(seed)

    X_tr, X_va, X_te = X[tr_idx], X[va_idx], X[te_idx]
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]

    # Scale
    x_scaler = StandardScaler().fit(X_tr)
    y_scaler = StandardScaler().fit(y_tr.reshape(-1,1))
    X_tr_s = x_scaler.transform(X_tr); X_va_s = x_scaler.transform(X_va); X_te_s = x_scaler.transform(X_te)
    y_tr_s = y_scaler.transform(y_tr.reshape(-1,1)).ravel()
    y_va_s = y_scaler.transform(y_va.reshape(-1,1)).ravel()

    Ttr = torch.from_numpy(X_tr_s).float().to(DEVICE)
    Tva = torch.from_numpy(X_va_s).float().to(DEVICE)
    Tte = torch.from_numpy(X_te_s).float().to(DEVICE)
    Ytr = torch.from_numpy(y_tr_s).float().to(DEVICE)
    Yva = torch.from_numpy(y_va_s).float().to(DEVICE)

    model = MLP(in_dim=X.shape[1], hidden=args.hidden).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, verbose=False)

    best_val = float("inf"); best_state = None
    history = {"train_rmse": [], "val_rmse": []}
    patience = args.patience

    for ep in range(1, args.epochs+1):
        model.train()
        perm = torch.randperm(Ttr.size(0), device=DEVICE)
        for i in range(0, Ttr.size(0), args.batch_size):
            idx = perm[i:i+args.batch_size]
            pred = model(Ttr[idx])
            loss = loss_fn(pred, Ytr[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            tr_pred_s = model(Ttr).cpu().numpy()
            va_pred_s = model(Tva).cpu().numpy()
        tr_pred = y_scaler.inverse_transform(tr_pred_s.reshape(-1,1)).ravel()
        va_pred = y_scaler.inverse_transform(va_pred_s.reshape(-1,1)).ravel()
        tr_rmse = rmse(y_tr, tr_pred); va_rmse = rmse(y_va, va_pred)
        history["train_rmse"].append(tr_rmse); history["val_rmse"].append(va_rmse)
        sched.step(va_rmse)
        if va_rmse < best_val - 1e-6:
            best_val, best_state, patience = va_rmse, model.state_dict(), args.patience
        else:
            patience -= 1
            if patience == 0: break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test predictions (same indices/order across all seeds!)
    model.eval()
    with torch.no_grad():
        te_pred_s = model(Tte).cpu().numpy()
    y_pred = y_scaler.inverse_transform(te_pred_s.reshape(-1,1)).ravel()

    # Metrics
    metrics = {
        "rmse": rmse(y_te, y_pred),
        "mse": mean_squared_error(y_te, y_pred),
        "mae": mean_absolute_error(y_te, y_pred),
        "r2": r2_score(y_te, y_pred),
    }
    if len(y_te) >= 3:
        metrics["pearson_r"]  = float(pearsonr(y_te, y_pred)[0])
        metrics["spearman_r"] = float(spearmanr(y_te, y_pred)[0])
    else:
        metrics["pearson_r"] = metrics["spearman_r"] = np.nan
    for d in (0.3, 0.5):
        metrics[f"acc@±{d}"] = float(np.mean(np.abs(y_te - y_pred) <= d))

    # Artifacts
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, f"loss_history_seed{seed}.csv"), index=False)
    plt.figure()
    plt.plot(history["train_rmse"], label="train RMSE")
    plt.plot(history["val_rmse"], label="val RMSE")
    plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.legend(); plt.title("Train/Val RMSE")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"loss_seed{seed}.png"), dpi=150); plt.close()

    plt.figure()
    plt.scatter(y_te, y_pred, s=12, alpha=0.6)
    lo, hi = float(min(y_te.min(), y_pred.min())), float(max(y_te.max(), y_pred.max()))
    plt.plot([lo,hi],[lo,hi])
    plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("Test: Pred vs True")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"scatter_seed{seed}.png"), dpi=150); plt.close()

    pd.DataFrame({"y_true": y_te, "y_pred": y_pred}).to_csv(
        os.path.join(out_dir, f"test_preds_seed{seed}.csv"), index=False
    )
    return metrics, y_te.copy(), y_pred

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--emb-dir", required=True)
    p.add_argument("--chembl-dir", required=True)       # points to chembl_built
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hf-model", default="ibm/biomed.sm.mv-te-84m")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--split", choices=["scaffold","random"], default="scaffold")
    p.add_argument("--split-seed", type=int, default=42, help="one fixed split for all seeds (critical for ensembling)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--reps", default="orig,proj_triplet,proj_ntxent")
    return p.parse_args()

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
            y = df["pchembl_median"].astype(float).values
        elif "standard_value" in df.columns:
            # fallback: standard_value assumed already pIC50-like per earlier pipeline
            y = df["standard_value"].astype(float).values
        else:
            raise ValueError("Could not find 'pchembl_median' or 'standard_value' in dataset.")

        smiles = df["std_smiles"].tolist()

        # Backbone + heads
        backbone = load_backbone(args.hf_model)
        in_dim = get_base_dim(backbone)
        heads = load_heads_for_target(tid, in_dim, args.emb_dir)

        for rep in [r.strip() for r in args.reps.split(",") if r.strip()]:
            print(f"\n-- rep {rep}")
            X = embed_smiles(backbone, heads, smiles, rep)

            # ---- FIXED split computed ONCE per rep ----
            if args.split == "scaffold":
                tr_idx, va_idx, te_idx = scaffold_split_indices(smiles, 0.7, 0.15, seed=args.split_seed)
            else:
                tr_idx, va_idx, te_idx = random_split_indices(len(smiles), 0.7, 0.15, seed=args.split_seed)

            rep_out = os.path.join(args.output_dir, f"target_{tid}", rep)
            os.makedirs(rep_out, exist_ok=True)

            seed_metrics = []
            all_preds = []
            y_test_ref = None

            for s in range(args.seeds):
                out_dir = os.path.join(rep_out, f"seed_{s}")
                m, y_te, y_pred = train_on_fixed_split(
                    X, y, tr_idx, va_idx, te_idx, args, seed=BASE_SEED+s, out_dir=out_dir
                )
                seed_metrics.append({"seed": s, **m})
                all_preds.append(y_pred)
                if y_test_ref is None:
                    y_test_ref = y_te

            # Ensemble: all test arrays now same length & order → safe to stack
            ens_pred = np.mean(np.vstack(all_preds), axis=0)
            ens_metrics = {
                "rmse": rmse(y_test_ref, ens_pred),
                "mse": mean_squared_error(y_test_ref, ens_pred),
                "mae": mean_absolute_error(y_test_ref, ens_pred),
                "r2": r2_score(y_test_ref, ens_pred),
                "pearson_r": float(pearsonr(y_test_ref, ens_pred)[0]) if len(y_test_ref)>=3 else np.nan,
                "spearman_r": float(spearmanr(y_test_ref, ens_pred)[0]) if len(y_test_ref)>=3 else np.nan,
                "acc@±0.3": float(np.mean(np.abs(y_test_ref - ens_pred) <= 0.3)),
                "acc@±0.5": float(np.mean(np.abs(y_test_ref - ens_pred) <= 0.5)),
            }
            print(f"   Ensemble → RMSE {ens_metrics['rmse']:.3f}  R² {ens_metrics['r2']:.3f}  acc@±0.5 {ens_metrics['acc@±0.5']:.3f}")

            # save artifacts
            pd.DataFrame(seed_metrics).to_csv(os.path.join(rep_out, "seed_metrics.csv"), index=False)
            pd.DataFrame({"y_true": y_test_ref, "y_pred_ensemble": ens_pred}).to_csv(
                os.path.join(rep_out, "test_preds_ensemble.csv"), index=False
            )
            plt.figure()
            plt.scatter(y_test_ref, ens_pred, s=12, alpha=0.6)
            lo, hi = float(min(y_test_ref.min(), ens_pred.min())), float(max(y_test_ref.max(), ens_pred.max()))
            plt.plot([lo, hi],[lo, hi]); plt.xlabel("True"); plt.ylabel("Pred (ensemble)")
            plt.title("Test: Pred vs True (ensemble)")
            plt.tight_layout(); plt.savefig(os.path.join(rep_out, "scatter_ensemble.png"), dpi=150); plt.close()

            summary.append({"target": tname, "tid": tid, "rep": rep, **ens_metrics})

    pd.DataFrame(summary).to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    print("\n✅ Done. Outputs in:", args.output_dir)

