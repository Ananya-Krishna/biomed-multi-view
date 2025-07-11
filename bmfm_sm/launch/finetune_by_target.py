import os
# disable Numba caching globally to avoid no-locator errors in UMAP/pynndescent
os.environ['NUMBA_DISABLE_JIT'] = '1'

import sys
import argparse
import random
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

# Try UMAP; fallback if unavailable
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel, LateFusionStrategy
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import Graph2dFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.image_finetune_dataset import ImageFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.text_finetune_dataset import TextFinetuneDataPipeline

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()
    def forward(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature
        mask = (~torch.eye(2 * B, device=z.device).bool()).float()
        sim = sim * mask
        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels], dim=0)
        return self.ce(sim, labels)

class PairDataset(Dataset):
    def __init__(self, pairs, emb_cache):
        self.pairs = pairs
        self.emb = emb_cache
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        return self.emb[a], self.emb[b]

class TripletDataset(Dataset):
    def __init__(self, pos_pairs, neg_pairs, emb_cache):
        self.pos = pos_pairs
        self.neg = neg_pairs
        self.emb = emb_cache
    def __len__(self): return len(self.pos)
    def __getitem__(self, idx):
        a, p = self.pos[idx]
        _, n = random.choice(self.neg)
        return self.emb[a], self.emb[p], self.emb[n]


def compute_distances(pairs, arr, idx_map, metric):
    d = []
    for a,b in pairs:
        ia, ib = idx_map[a], idx_map[b]
        if metric=='euclid':
            d.append(np.linalg.norm(arr[ia] - arr[ib]))
        else:
            d.append(cosine_distances(arr[[ia]], arr[[ib]])[0,0])
    return np.array(d)


def visualize(arr, smiles_list, pos_pairs, neg_pairs, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    idx_map = {s:i for i,s in enumerate(smiles_list)}
    # hist + means
    for m in ['euclid','cosine']:
        pos_d = compute_distances(pos_pairs, arr, idx_map, m)
        neg_d = compute_distances(neg_pairs, arr, idx_map, m)
        plt.figure(); plt.hist(pos_d,bins=50,alpha=0.5,label='Pos'); plt.hist(neg_d,bins=50,alpha=0.5,label='Neg')
        plt.legend(); plt.title(f"{tag} {m}"); plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"hist_{tag}_{m}.png")); plt.close()
        plt.figure(); plt.bar(['Pos','Neg'],[pos_d.mean(),neg_d.mean()]); plt.title(f"{tag} mean {m}"); plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"mean_{tag}_{m}.png")); plt.close()
    # PCA
    coords = PCA(n_components=2).fit_transform(arr)
    colors = ['tab:blue' if any(s in p for p in pos_pairs) else 'tab:orange' for s in smiles_list]
    plt.figure(); plt.scatter(coords[:,0],coords[:,1],c=colors,s=5,alpha=0.6); plt.title(f"PCA {tag}"); plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"pca_{tag}.png")); plt.close()
    # UMAP
    if UMAP_AVAILABLE:
        coords2 = umap.UMAP(n_components=2,random_state=42).fit_transform(arr)
        plt.figure(); plt.scatter(coords2[:,0],coords2[:,1],c=colors,s=5,alpha=0.6); plt.title(f"UMAP {tag}"); plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"umap_{tag}.png")); plt.close()
    else:
        print(f"⚠️ Skip UMAP {tag}", file=sys.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pos-csv',required=True)
    p.add_argument('--neg-csv',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--model',default='ibm/biomed.sm.mv-te-84m')
    p.add_argument('--embed-device',default='cpu')
    p.add_argument('--bs',type=int,default=32)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--val-split',type=float,default=0.1)
    p.add_argument('--temperature',type=float,default=0.5)
    p.add_argument('--margin',type=float,default=1.0)
    p.add_argument('--hidden',type=int,default=512)
    p.add_argument('--proj-dim',type=int,default=128)
    args = p.parse_args()

    print("🚀 START", file=sys.stdout)
    print("PWD:",os.getcwd(), file=sys.stdout)
    print("OUTDIR:",args.output_dir, file=sys.stdout)
    os.makedirs(args.output_dir,exist_ok=True)

    df_p = pd.read_csv(args.pos_csv)
    df_n = pd.read_csv(args.neg_csv)
    pos_data = [(r[0],r[2],r[4]) for r in df_p.itertuples(index=False,name=None)]
    neg_data = [(r[0],r[2],r[4]) for r in df_n.itertuples(index=False,name=None)]
    tids = sorted({t for *_,t in pos_data+neg_data})

    # global stats
    d_global = {k:{m:{'pos':[],'neg':[]} for m in ['euclid','cosine']} for k in ['orig','ntxent','triplet']}

    emb_cache = {}
    backbone = None

    for tid in tids:
        tgt = f"target_{tid}"; tdir = os.path.join(args.output_dir,tgt)
        os.makedirs(tdir,exist_ok=True)
        pd_pairs = [(a,b) for a,b,t in pos_data if t==tid]
        ng_pairs = [(a,b) for a,b,t in neg_data if t==tid]
        if not pd_pairs or not ng_pairs:
            print(f"Skip tid={tid}, missing pairs",file=sys.stdout)
            continue

        # load or init backbone
        if backbone is None:
            dev_e = torch.device(args.embed_device)
            backbone = SmallMoleculeMultiViewModel.from_pretrained(
                model_path=args.model,
                fusion_strategy=LateFusionStrategy.ATTENTIONAL,
                inference_mode=True,
                huggingface=True
            ).to(dev_e).eval()

        # compute embeddings for this target's SMILES
        for smi in {s for p in pd_pairs+ng_pairs for s in p}:
            if smi not in emb_cache:
                data = {}
                data.update(Graph2dFinetuneDataPipeline.smiles_to_graph_format(smi))
                data.update(TextFinetuneDataPipeline.smiles_to_text_format(smi))
                data.update(ImageFinetuneDataPipeline.smiles_to_image_format(smi))
                data['label'] = torch.zeros(1)
                for k,v in data.items():
                    if torch.is_tensor(v): data[k] = v.to(dev_e)
                out = backbone(data)
                emb_cache[smi] = (out[0] if isinstance(out,tuple) else out).squeeze().detach().cpu()
        torch.save(emb_cache, os.path.join(args.output_dir,'emb_cache.pt'))

        # build orig array
        smiles_list = list(emb_cache.keys())
        idx_map = {s:i for i,s in enumerate(smiles_list)}
        orig_arr = np.stack([emb_cache[s].numpy() for s in smiles_list])
        np.save(os.path.join(tdir,'orig.npy'),orig_arr)
        # record global stats
        for m in ['euclid','cosine']:
            d_global['orig'][m]['pos'].append(compute_distances(pd_pairs,orig_arr,idx_map,m))
            d_global['orig'][m]['neg'].append(compute_distances(ng_pairs,orig_arr,idx_map,m))

        # split train/val with fallback
        all_pairs = pd_pairs + ng_pairs
        labs = [1]*len(pd_pairs) + [0]*len(ng_pairs)
        try:
            tr_idx, va_idx = train_test_split(
                list(range(len(all_pairs))), test_size=args.val_split,
                stratify=labs, random_state=42
            )
        except ValueError:
            tr_idx, va_idx = train_test_split(
                list(range(len(all_pairs))), test_size=args.val_split,
                random_state=42
            )
        tr_p = [all_pairs[i] for i in tr_idx if labs[i]==1]
        tr_n = [all_pairs[i] for i in tr_idx if labs[i]==0]
        va_p = [all_pairs[i] for i in va_idx if labs[i]==1]
        va_n = [all_pairs[i] for i in va_idx if labs[i]==0]
        # skip if either set empty
        if not tr_p or not tr_n or not va_p or not va_n:
            print(f"Skip tid={tid}, insufficient train/val split",file=sys.stdout)
            continue

        # per-loss fine-tune
        for name, crit in [('ntxent', NTXentLoss(args.temperature)), ('triplet', nn.TripletMarginLoss(margin=args.margin))]:
            print(f"-- tid={tid} loss={name}",file=sys.stdout)
            ds_tr = TripletDataset(tr_p, tr_n, emb_cache) if name=='triplet' else PairDataset(tr_p, emb_cache)
            ds_va = TripletDataset(va_p, va_n, emb_cache) if name=='triplet' else PairDataset(va_p, emb_cache)
            dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True)
            dl_va = DataLoader(ds_va, batch_size=args.bs)
            dev_h = torch.device('cpu')  # use CPU to reduce GPU memory
            head = nn.Sequential(nn.Linear(orig_arr.shape[1], args.hidden), nn.ReLU(), nn.Linear(args.hidden, args.proj_dim)).to(dev_h)
            opt = torch.optim.Adam(head.parameters(), lr=args.lr)
            hist = []
            for ep in range(1, args.epochs+1):
                head.train(); Ltr=[]
                for batch in dl_tr:
                    opt.zero_grad()
                    if name=='triplet': a,p,n = batch; loss = crit(head(a), head(p), head(n))
                    else: a,b0 = batch; loss = crit(head(a), head(b0))
                    loss.backward(); opt.step(); Ltr.append(loss.item())
                head.eval(); Lva=[]
                with torch.no_grad():
                    for batch in dl_va:
                        if name=='triplet': a,p,n = batch; loss = crit(head(a), head(p), head(n))
                        else: a,b0 = batch; loss = crit(head(a), head(b0))
                        Lva.append(loss.item())
                hist.append((ep, np.mean(Ltr), np.mean(Lva)))
                print(f"tid={tid} {name} Ep{ep} tr={hist[-1][1]:.3f} va={hist[-1][2]:.3f}",file=sys.stdout)
            torch.save(head.state_dict(), os.path.join(tdir, f"head_{name}.pth"))
            pd.DataFrame(hist, columns=['epoch','train_loss','val_loss']).to_csv(os.path.join(tdir, f"hist_{name}.csv"), index=False)
            proj = np.stack([head(emb_cache[s]).detach().cpu().numpy() for s in smiles_list])
            np.save(os.path.join(tdir, f"proj_{name}.npy"), proj)
            for m in ['euclid','cosine']:
                d_global[name][m]['pos'].append(compute_distances(pd_pairs, proj, idx_map, m))
                d_global[name][m]['neg'].append(compute_distances(ng_pairs, proj, idx_map, m))
            visualize(orig_arr, smiles_list, pd_pairs, ng_pairs, os.path.join(tdir, 'vis'), 'orig')
            visualize(proj, smiles_list, pd_pairs, ng_pairs, os.path.join(tdir, 'vis'), name)
            del head; gc.collect()

    # aggregate global
    agg = os.path.join(args.output_dir, 'agg_vis'); os.makedirs(agg, exist_ok=True)
    for k in d_global:
        for m in ['euclid','cosine']:
            allp = np.concatenate(d_global[k][m]['pos'])
            alln = np.concatenate(d_global[k][m]['neg'])
            plt.figure(); plt.hist(allp,bins=50,alpha=0.5,label='Pos'); plt.hist(alln,bins=50,alpha=0.5,label='Neg'); plt.legend(); plt.title(f"{k} agg {m}"); plt.tight_layout(); plt.savefig(os.path.join(agg, f"hist_{k}_{m}.png")); plt.close()
            plt.figure(); plt.bar(['Pos','Neg'], [allp.mean(), alln.mean()]); plt.title(f"{k} agg mean {m}"); plt.tight_layout(); plt.savefig(os.path.join(agg, f"mean_{k}_{m}.png")); plt.close()
    print("DONE", file=sys.stdout)

if __name__=='__main__':
    main()
