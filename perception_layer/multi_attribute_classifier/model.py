"""
1.B — Multi-Attribute Classifier: Model
========================================

EfficientNet-B0 backbone with four task-specific heads:

    Head        Output          Loss
    ──────────────────────────────────────────────────────
    fit         (3,) logits     KLDivLoss / soft targets
    style       (4,) logits     KLDivLoss / soft targets
    weather     (1,) scalar     SmoothL1Loss or MSELoss
    formality   (1,) scalar     SmoothL1Loss or MSELoss

Category embedding is fused into the visual feature vector so the model
learns garment-type-specific attribute distributions.

Important:
    Fashionpedia filtered category IDs are expected to be 0..26.
    Therefore, num_categories defaults to 27.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class MultiHeadAttributeClassifier(nn.Module):
    """
    EfficientNet-B0 multi-head attribute classifier with category-conditioned fusion.

    Parameters
    ----------
    num_categories:
        Number of Fashionpedia filtered categories.
        For your current generator, this should be 27.

    category_embed_dim:
        Dimension of the learnable category embedding.

    hidden_dim:
        Shared representation size before task heads.

    dropout:
        Dropout used in the shared projection and regression heads.

    pretrained:
        Whether to load ImageNet-pretrained EfficientNet-B0 weights.
    """

    def __init__(
        self,
        num_categories: int = 27,
        category_embed_dim: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.30,
        pretrained: bool = True,
    ):
        super().__init__()

        self.num_categories = int(num_categories)

        # ── Backbone ──────────────────────────────────────────────────────────
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        image_feat_dim = backbone.classifier[1].in_features  # 1280

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # ── Category Embedding ────────────────────────────────────────────────
        self.category_embed = nn.Embedding(
            num_embeddings=self.num_categories,
            embedding_dim=category_embed_dim,
        )

        # Project category embedding to the same dimension as image features.
        self.category_projection = nn.Linear(
            category_embed_dim,
            image_feat_dim,
        )

        # Interaction-aware fusion:
        #   image features
        #   image - category projection
        #   image * category projection
        fusion_dim = image_feat_dim * 3

        self.shared_proj = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
        )

        # ── Classification Heads ──────────────────────────────────────────────
        self.head_fit = self._make_classifier(
            in_dim=hidden_dim,
            out_dim=3,
        )

        self.head_style = self._make_classifier(
            in_dim=hidden_dim,
            out_dim=4,
        )

        # ── Regression Heads ──────────────────────────────────────────────────
        # Output is already constrained to [0, 1] with Sigmoid.
        self.head_weather = self._make_regressor(hidden_dim)
        self.head_formality = self._make_regressor(hidden_dim)

    @staticmethod
    def _make_classifier(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(128, out_dim),
        )

    @staticmethod
    def _make_regressor(in_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        images: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        images:
            Tensor with shape (B, 3, 224, 224)

        category_ids:
            Tensor with shape (B,). Expected IDs: 0..num_categories-1.
            Category validation should happen in the Dataset, not here.

        Returns
        -------
        dict:
            {
                "fit":             (B, 3) logits,
                "style":           (B, 4) logits,
                "weather_warmth":  (B,) scalar in [0, 1],
                "formality_score": (B,) scalar in [0, 1],
            }
        """
        category_ids = category_ids.long()

        # Visual backbone
        image_features = self.features(images)
        image_features = self.avgpool(image_features)
        image_features = image_features.flatten(1)  # (B, 1280)

        # Category conditioning
        category_features = self.category_embed(category_ids)          # (B, category_embed_dim)
        category_projected = self.category_projection(category_features)  # (B, 1280)

        # Interaction fusion
        fused = torch.cat(
            [
                image_features,
                image_features - category_projected,
                image_features * category_projected,
            ],
            dim=1,
        )

        shared = self.shared_proj(fused)

        return {
            "fit": self.head_fit(shared),
            "style": self.head_style(shared),

            # Higher = warmer clothing / more suitable for cold weather.
            "weather_warmth": self.head_weather(shared).squeeze(1),

            # Higher = more formal.
            "formality_score": self.head_formality(shared).squeeze(1),
        }