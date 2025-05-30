import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel, LateFusionStrategy, Graph2dFinetuneDataPipeline
from bmfm_sm.core.data_modules.namespace import Modality
from bmfm_sm.api.dataset_registry import DatasetRegistry

class NTXentLoss(nn.Module):
    """Simple NT-Xent (InfoNCE) for a batch of pairs."""
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=2)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        # z1, z2: (B, D)
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                    # (2B, D)
        sim = torch.matmul(z, z.T) / self.temperature     # (2B, 2B)
        # mask out self‐sims
        mask = (~torch.eye(2 * B, device=z.device).bool()).float()
        sim = sim * mask
        # labels: positives are offset by B
        labels = torch.arange(B, device=z.device)
        labels = torch.cat([labels + B, labels], dim=0)   # (2B,)
        loss = nn.CrossEntropyLoss()(sim, labels)
        return loss

@click.command()
@click.option("--model-path", default="ibm/biomed.sm.mv-te-84m",
              help="HuggingFace ID of the pretrained MV model")
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--lr", default=1e-4, help="Learning rate for MLP head")
@click.option("--epochs", default=5, help="Number of epochs")
@click.argument("pos_csv",   type=click.Path(exists=True))
@click.argument("neg_csv",   type=click.Path(exists=True))

def main(model_path, batch_size, lr, epochs, pos_csv, neg_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = SmallMoleculeMultiViewModel.from_pretrained(
        model_path=model_path,
        fusion_strategy=LateFusionStrategy.CONCAT,
        inference_mode=False,
        huggingface=True,
    )

    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device)


    in_dim = backbone.get_embed_dim()
    print(f"Embedding dimension is {in_dim}")

    proj_head = nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    ).to(device)

    optimizer = torch.optim.Adam(proj_head.parameters(), lr=lr)
    criterion = NTXentLoss(temperature=0.5)

    class MMPDataset(Dataset):
        def __init__(self, pos_csv, neg_csv):
            df1 = pd.read_csv(pos_csv)
            df2 = pd.read_csv(neg_csv)
            pos_pairs = list(zip(df1.iloc[:,0], df1.iloc[:,2]))
            neg_pairs = list(zip(df2.iloc[:,0], df2.iloc[:,2]))
        
            def is_graph_valid(smi):
                try:
                    g = Graph2dFinetuneDataPipeline.smiles_to_graph_format(smi)
                    return (g is not None) and hasattr(g, 'x')
                except Exception:
                    return False

            # keep only pairs where smiles yield a graph
            self.pairs = [
                (a,b)
                for (a,b) in pos_pairs + neg_pairs
                if is_graph_valid(a) and is_graph_valid(b)
            ]
            

        def __len__(self): return len(self.pairs)
        def __getitem__(self, i):
            a, b = self.pairs[i]
            return {"smiles_A": a, "smiles_B": b}

    loader = DataLoader(MMPDataset(pos_csv, neg_csv),batch_size=batch_size, shuffle=True)


    backbone.eval()
    for epoch in range(epochs):
        total_loss = 0.0
        text_encoder = backbone.model_text
        for batch in loader:
            smiles_A = batch["smiles_A"]
            smiles_B = batch["smiles_B"]
            # get frozen embeddings
            with torch.no_grad():
                eA = SmallMoleculeMultiViewModel.get_embeddings(
                    smiles=smiles_A,
                    fusion_strategy=LateFusionStrategy.CONCAT,
                    pretrained_model=backbone,                  
                    huggingface=True,                           
                ).to(device)

                eB = SmallMoleculeMultiViewModel.get_embeddings(
                    smiles=smiles_B,
                    fusion_strategy=LateFusionStrategy.CONCAT,
                    pretrained_model=backbone,
                    huggingface=True,
                ).to(device)

            # project
            z1 = proj_head(eA)
            z2 = proj_head(eB)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch} — avg loss: {total_loss/len(loader):.4f}")

if __name__ == "__main__":
    main()

