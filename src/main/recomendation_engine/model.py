import torch
import torch.nn as nn
import torch.nn.functional as F


class OutfitEmbeddingTransformer(nn.Module):
    def __init__(self, embed_dim=512, projection_dim=128, num_categories=50, num_heads=8, dropout=0.3):
        super().__init__()
        self.category_embed = nn.Embedding(num_categories, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, projection_dim)
        )

    def forward(self, item_vectors, category_ids):
        # item_vectors: (Batch, N, 512) | category_ids: (Batch, N)
        cat_embs = self.category_embed(category_ids)
        x = item_vectors + cat_embs

        x = self.transformer(x)
        outfit_vec = x.mean(dim=1)

        embedding = self.projection(outfit_vec)
        return F.normalize(embedding, p=2, dim=1)