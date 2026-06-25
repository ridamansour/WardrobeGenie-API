"""
1.B — Multi-Attribute Classifier: Inference
============================================
Drop-in replacement for the CLIP-based extract_attributes() function
described in the project spec, but uses the trained EfficientNet-B0
model instead of CLIP at inference time — much faster on-device.

Usage
-----
    from 1b_inference import AttributePredictor

    predictor = AttributePredictor("runs/1b/best_model.pt")
    result    = predictor.predict(pil_image, category_id=1)
    # {
    #   "fit":             [0.05, 0.85, 0.10],   # slim / regular / oversized
    #   "style":           [0.02, 0.90, 0.05, 0.03],
    #   "weather_warmth":  0.78,
    #   "formality_score": 0.22,
    # }
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from perception_layer.multi_attribute_classifier.old_model.model import MultiHeadAttributeClassifier

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std =[0.26862954, 0.26130258, 0.27577711],
    ),
])

FIT_LABELS   = ["slim", "regular", "oversized"]
STYLE_LABELS = ["formal", "casual", "athletic", "streetwear"]


class AttributePredictor:
    """
    Parameters
    ----------
    checkpoint_path : str
        Path to a .pt checkpoint saved by train.py.
    device : str | None
        "cuda" / "cpu" / None (auto-detect).
    """

    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        args = ckpt.get("args", {})

        self.model = MultiHeadAttributeClassifier(
            num_categories     = args.get("num_categories", 46),
            category_embed_dim = 64,
        ).to(self.device)
        self.model.load_state_dict(ckpt)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image, category_id: int = 0) -> dict:
        """
        Parameters
        ----------
        image       : PIL Image (ideally a cropped garment from 1.A)
        category_id : Fashionpedia category int

        Returns
        -------
        dict matching the project spec schema
        """
        img_tensor = _TRANSFORM(image).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        cat_tensor = torch.tensor([category_id], dtype=torch.long).to(self.device)

        out = self.model(img_tensor, cat_tensor)

        fit_probs   = F.softmax(out["fit"],   dim=-1).cpu().squeeze(0).tolist()
        style_probs = F.softmax(out["style"], dim=-1).cpu().squeeze(0).tolist()

        return {
            "fit":             fit_probs,            # [slim_p, regular_p, oversized_p]
            "style":           style_probs,          # [formal_p, casual_p, athletic_p, streetwear_p]
            "weather_warmth":  out["weather_warmth"].item(),
            "formality_score": out["formality_score"].item(),
            # Human-readable argmax labels
            "fit_label":       FIT_LABELS[fit_probs.index(max(fit_probs))],
            "style_label":     STYLE_LABELS[style_probs.index(max(style_probs))],
        }

if __name__ == "__main__":
    predictor = AttributePredictor("runs/1b/best_model.pt")
    image = Image.open("../../examples/ex5.jpeg")
    result = predictor.predict(image, category_id=1)
    print(result)