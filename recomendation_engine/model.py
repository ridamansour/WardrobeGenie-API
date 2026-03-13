import torch
import torch.nn as nn
import torch.nn.functional as F


class OutfitEmbeddingTransformer(nn.Module):
    """The Stylist: A Set-Transformer that encodes outfit harmony into a 128-dim space."""

    def __init__(self, embed_dim=512, projection_dim=128, num_categories=50, num_heads=8, dropout=0.3):
        super().__init__()
        # Category embeddings to provide semantic context to the visual vectors
        self.category_embed = nn.Embedding(num_categories, embed_dim)

        # Transformer Encoder to capture cross-item relationships (e.g., how the shoes match the top)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Projection head to the normalized "Vibe Space"
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, projection_dim)
        )

    def forward(self, item_vectors, category_ids):
        # item_vectors: (Batch, N, 512) | category_ids: (Batch, N)

        # 1. Add category context to visual embeddings
        cat_embs = self.category_embed(category_ids)
        x = item_vectors + cat_embs

        # 2. Contextualize items through attention layers
        x = self.transformer(x)

        # 3. Global Pooling: Mean of all items to represent the "Outfit Vector"
        outfit_vec = x.mean(dim=1)

        # 4. Project and Normalize (crucial for Triplet Loss distances)
        embedding = self.projection(outfit_vec)
        return F.normalize(embedding, p=2, dim=1)