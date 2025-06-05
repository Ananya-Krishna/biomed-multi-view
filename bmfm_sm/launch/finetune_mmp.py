import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

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
        inference_mode=True,  # Set to True since we're only using it for inference
        huggingface=True,
    )
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

    # 3) First, collect all unique SMILES from both datasets
    print("Reading CSV files...")
    df_pos = pd.read_csv(pos_csv)
    df_neg = pd.read_csv(neg_csv)
    
    # Extract SMILES pairs
    pos_pairs = list(zip(df_pos.iloc[:, 0], df_pos.iloc[:, 2]))
    neg_pairs = list(zip(df_neg.iloc[:, 0], df_neg.iloc[:, 2]))
    all_pairs = pos_pairs + neg_pairs
    
    # Get unique SMILES across all pairs
    unique_smiles = set()
    for a, b in all_pairs:
        unique_smiles.add(a)
        unique_smiles.add(b)
    
    unique_smiles = list(unique_smiles)
    print(f"Found {len(unique_smiles)} unique molecules")
    
    # 4) Precompute embeddings for all unique SMILES (only once)
    print("Precomputing embeddings for all molecules (this may take a while)...")
    
    # Use a dictionary to map SMILES to their embeddings
    embedding_cache = {}
    
    with torch.no_grad():
        for smiles in tqdm(unique_smiles, desc="Computing embeddings"):
            embedding_cache[smiles] = get_embedding(smiles, backbone).cpu()
    
    # Infer embedding dimension from the cached embeddings
    embedding_dim = next(iter(embedding_cache.values())).size(-1)
    print(f"Embedding dimension: {embedding_dim}")
    
    # 5) Build projection head on top of embeddings
    proj_head = nn.Sequential(
        nn.Linear(embedding_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    ).to(device)

    optimizer = torch.optim.Adam(proj_head.parameters(), lr=lr)
    criterion = NTXentLoss(temperature=0.5)
    
    # 6) Create dataset using precomputed embeddings
    class CachedEmbeddingDataset(Dataset):
        def __init__(self, pairs, labels, embedding_cache):
            self.pairs = pairs
            self.labels = labels
            self.embedding_cache = embedding_cache
            
        def __len__(self):
            return len(self.pairs)
            
        def __getitem__(self, idx):
            smiles_a, smiles_b = self.pairs[idx]
            label = self.labels[idx]
            
            emb_a = self.embedding_cache[smiles_a]
            emb_b = self.embedding_cache[smiles_b]
            
            return {
                'emb_a': emb_a,
                'emb_b': emb_b,
                'label': label,
                'smiles_a': smiles_a,
                'smiles_b': smiles_b
            }
    
    # Create labels for pairs
    pair_labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    
    # Create dataset with cached embeddings
    dataset = CachedEmbeddingDataset(all_pairs, pair_labels, embedding_cache)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 7) Train the projection head only
    print(f"Training projection head for {epochs} epochs...")
    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            # Get embeddings from cache
            embeddings_A = batch['emb_a'].to(device)
            embeddings_B = batch['emb_b'].to(device)
            labels = batch['label']
            
            # Project embeddings
            z1 = proj_head(embeddings_A)
            z2 = proj_head(embeddings_B)
            
            # Compute contrastive loss
            loss = criterion(z1, z2)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} — avg loss: {avg_loss:.4f}")
    
    # 8) Save the projection head and embeddings
    model_path = os.path.join(output_dir, "mmp_projection_head.pth")
    torch.save(proj_head.state_dict(), model_path)
    print(f"Training complete. Projection head saved to {model_path}")
    
    # 9) Generate and save both original and projected embeddings
    print("Saving original and projected embeddings...")
    
    # Convert embedding cache to arrays for saving
    smiles_list = list(embedding_cache.keys())
    original_embeddings = np.stack([embedding_cache[s].numpy() for s in smiles_list])
    
    # Generate projected embeddings
    projected_embeddings = []
    with torch.no_grad():
        for smiles in tqdm(smiles_list, desc="Generating projected embeddings"):
            emb = embedding_cache[smiles].to(device)
            proj_emb = proj_head(emb).cpu().numpy()
            projected_embeddings.append(proj_emb)
    
    projected_embeddings = np.array(projected_embeddings)
    
    # Save embeddings and SMILES
    np.save(os.path.join(output_dir, "original_embeddings.npy"), original_embeddings)
    np.save(os.path.join(output_dir, "projected_embeddings.npy"), projected_embeddings)
    
    # Save SMILES for reference
    with open(os.path.join(output_dir, "embedding_smiles.txt"), "w") as f:
        for smiles in smiles_list:
            f.write(f"{smiles}\n")
    
    print(f"Embeddings saved to {output_dir}")
    print(f"Original embeddings shape: {original_embeddings.shape}")
    print(f"Projected embeddings shape: {projected_embeddings.shape}")


if __name__ == "__main__":
    main()
