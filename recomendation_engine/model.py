import torch
import torch.nn as nn
import torch.nn.functional as F


class Gatekeeper:
    """Filters wardrobe items based on hard constraints and semantic similarity."""

    def __init__(self, wardrobe_db):
        self.wardrobe = wardrobe_db

    def filter_and_rank(self, query_vec, filters, context, top_k=50):
        candidates = []
        for item in self.wardrobe:
            # Hard Filters: Formality & Weather
            if item["metrics"]["formality"] < filters.get("min_formality", 0.0):
                continue

            temp = context.get("temperature", 20)
            # Prevent parkas in summer or tees in a blizzard
            if temp < 10 and item["metrics"]["weather_warmth"] > 0.7: continue
            if temp > 28 and item["metrics"]["weather_warmth"] < 0.3: continue

            # Semantic Similarity
            item_vec = torch.tensor(item["embedding"]).unsqueeze(0)
            similarity = F.cosine_similarity(item_vec, query_vec).item()
            candidates.append((similarity, item))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in candidates[:top_k]]


import torch
import torch.nn as nn

class OutfitSetTransformer(nn.Module):
    def __init__(self, embed_dim=512, num_categories=10, num_heads=4, dropout=0.3):
        super().__init__()
        self.category_embed = nn.Embedding(num_categories, embed_dim)

        # Self-attention allows every item to "look" at every other item
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),  # Added as requested
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, item_vectors, category_ids):
        # item_vectors: (Batch, N, 512) | category_ids: (Batch, N)
        cat_embs = self.category_embed(category_ids)
        x = item_vectors + cat_embs

        # attn_out: (Batch, N, 512)
        attn_out, _ = self.attention(x, x, x)

        # Mean pooling to get a single 'vibe' vector for the whole outfit
        outfit_representation = attn_out.mean(dim=1)
        return self.ffn(outfit_representation)


class PersonalPreferenceBandit:
    """Adjusts α/β weights based on user interaction (Reinforcement-lite)."""

    def __init__(self, alpha=0.6, beta=0.4, lr=0.05):
        self.alpha = alpha
        self.beta = beta
        self.lr = lr

    def update(self, liked, comp_score, rel_score):
        reward = 1 if liked else -1
        # Nudge weights based on which component contributed to the 'liked' state
        self.alpha += reward * (comp_score - 0.5) * self.lr
        self.beta += reward * (rel_score - 0.5) * self.lr

        # Clamp and Normalize
        self.alpha, self.beta = max(0.1, self.alpha), max(0.1, self.beta)
        total = self.alpha + self.beta
        self.alpha, self.beta = self.alpha / total, self.beta / total