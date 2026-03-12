import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import CLIPVisionModelWithProjection


class StudentEncoder(nn.Module):
    """MobileNetV3-Small: High speed, low memory for on-device embedding."""

    def __init__(self, embed_dim=512):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        # Update the classifier head
        # MobileNetV3-Small features output 576 channels before the final pooling
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, embed_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(x, p=2, dim=1)


class TeacherEncoder(nn.Module):
    """Frozen CLIP ViT-B/32 used as the knowledge source."""

    def __init__(self):
        super().__init__()
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return F.normalize(outputs.image_embeds, p=2, dim=1)