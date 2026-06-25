import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import CLIPVisionModelWithProjection


class StudentEncoder(nn.Module):
    """
    MobileNetV3-Small optimized for high-speed, low-memory
    on-device feature extraction and embedding generation.
    """

    def __init__(self, embed_dim=512):
        super().__init__()
        # Load the pre-trained MobileNetV3 backbone
        self.backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

        # Update the classifier head to map to the embedding dimension
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, embed_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        # L2 normalization to project embeddings onto a unit hypersphere
        return F.normalize(x, p=2, dim=1)


class TeacherEncoder(nn.Module):
    """
    Frozen FashionCLIP ViT-B/32 used as the domain-specific
    knowledge source for distillation.
    """

    def __init__(self):
        super().__init__()
        # Load specialized domain weights from Hugging Face
        self.model = CLIPVisionModelWithProjection.from_pretrained("patrickjohncyh/fashion-clip")

        # Freeze all teacher parameters to save memory and compute
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        # L2 normalization to align representation space with the student
        return F.normalize(outputs.image_embeds, p=2, dim=1)


if __name__ == "__main__":
    # Quick sanity check for tensor dimensions
    print("Testing model dimensions...")

    # Simulate a batch of 2 RGB images cropped to 224x224
    dummy_input = torch.randn(2, 3, 224, 224)

    student = StudentEncoder()
    teacher = TeacherEncoder()

    with torch.no_grad():
        student_out = student(dummy_input)
        teacher_out = teacher(dummy_input)

    print(f"Student output shape: {student_out.shape} (Expected: [2, 512])")
    print(f"Teacher output shape: {teacher_out.shape} (Expected: [2, 512])")

    assert student_out.shape == teacher_out.shape == (2, 512), "Dimension mismatch!"
    print("Success! Dimensions match perfectly.")