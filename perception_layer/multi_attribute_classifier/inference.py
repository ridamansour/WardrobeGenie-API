"""
1.B — Multi-Attribute Classifier: Inference
============================================
Drop-in replacement for the CLIP-based extract_attributes() function
described in the project spec, but uses the trained EfficientNet-B0
model instead of CLIP at inference time — much faster on-device.

API contract
------------
    predictor = AttributePredictor("attribute_predictor/1b/best_model.pt")
    result = predictor.predict(pil_image, category_id=1)

Returns
-------
    {
      "fit":             [float, float, float],
      "style":           [float, float, float, float],
      "weather_warmth":  float,
      "formality_score": float,
      "fit_label":       str,
      "style_label":     str,
    }

Important:
    The model was trained with ImageNet normalization because the student
    backbone is EfficientNet-B0.

    Fashionpedia filtered category IDs are expected to be 0..26.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms


from perception_layer.multi_attribute_classifier.model import MultiHeadAttributeClassifier


# EfficientNet-B0 / ImageNet normalization.
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

FIT_LABELS = ["slim", "regular", "oversized"]
STYLE_LABELS = ["formal", "casual", "athletic", "streetwear"]


def _get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _safe_torch_load(path: str, device: str):
    """
    Compatible with both newer and older PyTorch versions.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class AttributePredictor:
    """
    Parameters
    ----------
    checkpoint_path:
        Path to a .pt checkpoint saved by train.py.

        Supported checkpoint formats:
            1. Raw model state_dict:
                best_model.pt

            2. Full checkpoint dictionary:
                {
                    "model": state_dict,
                    "config": {...}
                }

    device:
        "cuda" / "mps" / "cpu" / None.
        If None, auto-detects the best available device.

    num_categories:
        Number of Fashionpedia filtered categories.
        Default is 27 because your COCO generator remaps categories to 0..26.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        num_categories: int = 27,
    ):
        self.device = device or _get_best_device()

        ckpt = _safe_torch_load(checkpoint_path, self.device)

        # Default values matching the final model.py.
        model_config = {
            "num_categories": num_categories,
            "category_embed_dim": 64,
            "hidden_dim": 512,
            "dropout": 0.30,
            "pretrained": False,  # Avoid downloading ImageNet weights during API startup.
        }

        # Case 1: full checkpoint from train.py
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]

            # The updated train.py stores config under "config".
            # Older variants may store it under "args".
            saved_config = ckpt.get("config", ckpt.get("args", {}))

            if isinstance(saved_config, dict):
                model_config["num_categories"] = int(
                    saved_config.get("num_categories", model_config["num_categories"])
                )
                model_config["dropout"] = float(
                    saved_config.get("dropout", model_config["dropout"])
                )

        # Case 2: raw state_dict from best_model.pt
        else:
            state_dict = ckpt

        self.num_categories = int(model_config["num_categories"])

        self.model = MultiHeadAttributeClassifier(
            num_categories=self.num_categories,
            category_embed_dim=model_config["category_embed_dim"],
            hidden_dim=model_config["hidden_dim"],
            dropout=model_config["dropout"],
            pretrained=model_config["pretrained"],
        ).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _validate_category_id(self, category_id: int) -> int:
        category_id = int(category_id)

        if category_id < 0 or category_id >= self.num_categories:
            raise ValueError(
                f"category_id must be in [0, {self.num_categories - 1}], "
                f"got {category_id}"
            )

        return category_id

    @torch.no_grad()
    def predict(self, image: Image.Image, category_id: int = 0) -> dict:
        """
        Parameters
        ----------
        image:
            PIL Image. Ideally a cropped garment from 1.A.

        category_id:
            Fashionpedia filtered category ID.
            Expected range: 0..26.

        Returns
        -------
        dict matching the project spec schema.
        """
        category_id = self._validate_category_id(category_id)

        image = ImageOps.exif_transpose(image).convert("RGB")

        img_tensor = _TRANSFORM(image).unsqueeze(0).to(self.device)
        cat_tensor = torch.tensor(
            [category_id],
            dtype=torch.long,
            device=self.device,
        )

        out = self.model(img_tensor, cat_tensor)

        fit_probs_tensor = F.softmax(out["fit"], dim=-1)[0].detach().cpu()
        style_probs_tensor = F.softmax(out["style"], dim=-1)[0].detach().cpu()

        fit_probs = [float(v) for v in fit_probs_tensor.tolist()]
        style_probs = [float(v) for v in style_probs_tensor.tolist()]

        weather_warmth = float(
            out["weather_warmth"][0]
            .detach()
            .clamp(0.0, 1.0)
            .cpu()
            .item()
        )

        formality_score = float(
            out["formality_score"][0]
            .detach()
            .clamp(0.0, 1.0)
            .cpu()
            .item()
        )

        fit_idx = int(fit_probs_tensor.argmax().item())
        style_idx = int(style_probs_tensor.argmax().item())

        return {
            "fit": fit_probs,
            "style": style_probs,
            "weather_warmth": weather_warmth,
            "formality_score": formality_score,
            "fit_label": FIT_LABELS[fit_idx],
            "style_label": STYLE_LABELS[style_idx],
        }


if __name__ == "__main__":
    predictor = AttributePredictor(
        "../../models/attribute_predictor/1b/best_model.pt",
        num_categories=27,
    )

    image = Image.open("../../examples/ex5.jpeg")
    result = predictor.predict(image, category_id=1)

    print(result)