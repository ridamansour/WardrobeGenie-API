"""
1.B — Multi-Attribute Classifier: Model
========================================
EfficientNet-B0 backbone with four task-specific heads:

    Head        Output          Loss
    ──────────────────────────────────────────────────────
    fit         (3,)  logits    KLDivLoss  (soft targets)
    style       (4,)  logits    KLDivLoss  (soft targets)
    weather     (1,)  scalar    MSELoss    (regression)
    formality   (1,)  scalar    MSELoss    (regression)

Category embedding is fused into the feature vector so the model
learns garment-type-specific attribute distributions (e.g. shoes
should never be "oversized").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MultiHeadAttributeClassifier(nn.Module):
    """
    Parameters
    ----------
    num_categories : int
        Number of Fashionpedia categories (default 46).
    category_embed_dim : int
        Dimension of the learnable category embedding.
    dropout : float
        Dropout applied before each head.
    """

    def __init__(
        self,
        num_categories:    int   = 46,
        category_embed_dim: int  = 64,
        dropout:           float = 0.3,
    ):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────────
        backbone   = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats   = backbone.classifier[1].in_features   # 1280

        # Strip the original classifier; keep everything up to the pooled features
        self.features   = backbone.features
        self.avgpool    = backbone.avgpool

        # ── Category Embedding ────────────────────────────────────────────────
        self.category_embed = nn.Embedding(num_categories, category_embed_dim)

        # ── Shared Projection (visual + category → fused) ─────────────────────
        fused_dim = in_feats + category_embed_dim
        self.shared_proj = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # ── Task Heads ────────────────────────────────────────────────────────
        # Classification heads return logits (softmax / KLDiv applied in loss)
        self.head_fit       = self._make_head(512, 3)    # slim / regular / oversized
        self.head_style     = self._make_head(512, 4)    # formal / casual / athletic / streetwear

        # Regression heads return a single unbounded scalar (sigmoid-squashed in loss)
        self.head_weather   = self._make_regressor(512)  # 0=cold  → 1=warm
        self.head_formality = self._make_regressor(512)  # 0=pjs   → 1=tux

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_head(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    @staticmethod
    def _make_regressor(in_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),   # output ∈ [0, 1]
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        images:       torch.Tensor,    # (B, 3, 224, 224)
        category_ids: torch.Tensor,    # (B,)  long
    ) -> dict[str, torch.Tensor]:
        # Visual backbone
        x = self.features(images)
        x = self.avgpool(x)
        x = x.flatten(1)               # (B, 1280)

        # Category conditioning
        cat = self.category_embed(category_ids)   # (B, 64)
        x   = torch.cat([x, cat], dim=-1)         # (B, 1344)

        # Shared projection
        x = self.shared_proj(x)                   # (B, 512)

        return {
            "fit":             self.head_fit(x),            # (B, 3)  logits
            "style":           self.head_style(x),          # (B, 4)  logits
            "weather_warmth":  self.head_weather(x).squeeze(1),   # (B,)
            "formality_score": self.head_formality(x).squeeze(1), # (B,)
        }
