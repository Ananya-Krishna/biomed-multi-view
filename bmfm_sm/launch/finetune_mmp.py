import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel, LateFusionStrategy
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import Graph2dFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.image_finetune_dataset import ImageFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.text_finetune_dataset import TextFinetuneDataPipeline


class NTXentLoss(nn.Module):
    """Simple NT-Xent (InfoNCE) for a batch of pairs."""
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        # z1, z2: (B, D)
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                    # (2B, D)
        sim = torch.matmul(z, z.T) / self.temperature     # (2B, 2B)
        # mask out self-sims
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
@click.option("--output-dir", default="mmp_output", help="Directory to save outputs")
@click.argument("pos_csv",   type=click.Path(exists=True))
@click.argument("neg_csv",   type=click.Path(exists=True))
def main(model_path, batch_size, lr, epochs, output_dir, pos_csv, neg_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1) Load the pretrained multimodal backbone
    print("Loading pretrained model...")
    backbone = SmallMoleculeMultiViewModel.from_pretrained(
        model_path=model_path,
        fusion_strategy=LateFusionStrategy.ATTENTIONAL,  # Using the default strategy
        inference_mode=False,
        huggingface=True,
    )
    # Freeze the backbone
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device)
    backbone.eval()

    # 2) Define a helper function to get embeddings that adds a dummy label
    def get_embedding(smiles, model):
        joint_dict = {}
        joint_dict.update(Graph2dFinetuneDataPipeline.smiles_to_graph_format(smiles))
        joint_dict.update(TextFinetuneDataPipeline.smiles_to_text_format(smiles))
        joint_dict.update(ImageFinetuneDataPipeline.smiles_to_image_format(smiles))
        
        # Add a dummy label to avoid KeyError
        joint_dict['label'] = torch.zeros(1).to(device)
        
        # The model returns a tuple, get the first element (embeddings)
        output = model.forward(joint_dict)
        if isinstance(output, tuple):
            return output[0].squeeze()
        else:
            return output.squeeze()

    # Infer embedding dimension dynamically via a dummy pass
    with torch.no_grad():
        # Get a full multimodal embedding for a dummy SMILES
        dummy_emb = get_embedding("CC", backbone)
        embedding_dim = dummy_emb.size(-1)

    print(f"Inferred embedding dimension: {embedding_dim}")

    # 3) Build projection head on top of embeddings
    proj_head = nn.Sequential(
        nn.Linear(embedding_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    ).to(device)

    optimizer = torch.optim.Adam(proj_head.parameters(), lr=lr)
    criterion = NTXentLoss(temperature=0.5)

    class MMPDataset(Dataset):
        """Load all SMILES pairs without any graph filtering."""
        def __init__(self, pos_csv, neg_csv):
            df1 = pd.read_csv(pos_csv)
            df2 = pd.read_csv(neg_csv)
            pos_pairs = list(zip(df1.iloc[:, 0], df1.iloc[:, 2]))
            neg_pairs = list(zip(df2.iloc[:, 0], df2.iloc[:, 2]))
            self.pairs = pos_pairs + neg_pairs
            # Store which pairs are positive (1) vs negative (0)
            self.labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, i):
            a, b = self.pairs[i]
            return {"smiles_A": a, "smiles_B": b, "label": self.labels[i]}

    loader = DataLoader(
        MMPDataset(pos_csv, neg_csv),
        batch_size=batch_size,
        shuffle=True,
    )

    print(f"Starting training for {epochs} epochs...")
    # 4) Training loop: full multimodal finetuning
    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            smiles_A = batch["smiles_A"]
            smiles_B = batch["smiles_B"]
            labels = batch["label"]

            # Get embeddings from the frozen backbone
            with torch.no_grad():
                # Get embeddings for batch A
                embeddings_A = torch.stack([
                    get_embedding(smiles, backbone).to(device)
                    for smiles in tqdm(smiles_A, desc="Processing A", leave=False)
                ])
                
                # Get embeddings for batch B
                embeddings_B = torch.stack([
                    get_embedding(smiles, backbone).to(device)
                    for smiles in tqdm(smiles_B, desc="Processing B", leave=False)
                ])

            # Project & compute contrastive loss
            z1 = proj_head(embeddings_A)  # (B, 128)
            z2 = proj_head(embeddings_B)  # (B, 128)
            
            # Apply contrastive loss
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} — avg loss: {avg_loss:.4f}")

    # Save the projection head
    model_path = os.path.join(output_dir, "mmp_projection_head.pth")
    torch.save(proj_head.state_dict(), model_path)
    print(f"Training complete. Projection head saved to {model_path}")
    
    # Generate and save embeddings for a sample of molecules
    print("Generating embeddings for sample molecules...")
    
    # Create a dataset reader to get all unique SMILES
    df_pos = pd.read_csv(pos_csv)
    df_neg = pd.read_csv(neg_csv)
    
    # Get unique SMILES from both files
    all_smiles = set()
    for df in [df_pos, df_neg]:
        all_smiles.update(df.iloc[:, 0])
        all_smiles.update(df.iloc[:, 2])
    
    all_smiles = list(all_smiles)
    print(f"Found {len(all_smiles)} unique molecules")
    
    # Limit to 1000 molecules if there are too many
    if len(all_smiles) > 1000:
        all_smiles = all_smiles[:1000]
        print(f"Limiting to first 1000 molecules")
    
    # Generate embeddings
    original_embeddings = []
    projected_embeddings = []
    
    with torch.no_grad():
        for smiles in tqdm(all_smiles, desc="Generating embeddings"):
            # Get original embedding
            emb = get_embedding(smiles, backbone).cpu()
            original_embeddings.append(emb.numpy())
            
            # Get projected embedding
            proj_emb = proj_head(emb.to(device)).cpu().numpy()
            projected_embeddings.append(proj_emb)
    
    # Convert to numpy arrays
    original_embeddings = np.array(original_embeddings)
    projected_embeddings = np.array(projected_embeddings)
    
    # Save embeddings and SMILES
    np.save(os.path.join(output_dir, "original_embeddings.npy"), original_embeddings)
    np.save(os.path.join(output_dir, "projected_embeddings.npy"), projected_embeddings)
    
    # Save SMILES for reference
    with open(os.path.join(output_dir, "embedding_smiles.txt"), "w") as f:
        for smiles in all_smiles:
            f.write(f"{smiles}\n")
    
    print(f"Embeddings saved to {output_dir}")
    print(f"Original embeddings shape: {original_embeddings.shape}")
    print(f"Projected embeddings shape: {projected_embeddings.shape}")


if __name__ == "__main__":
    main()
