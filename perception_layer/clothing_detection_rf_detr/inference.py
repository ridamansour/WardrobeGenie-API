"""
inference.py

Run garment detection using RF-DETR Nano.

Outputs:
- bbox
- mask (optional)
- category
- confidence
- cropped garment image
"""

from pathlib import Path
from typing import List, Dict, Any
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from perception_layer.clothing_detection_rf_detr.model import load_model, CLASS_NAMES, DEVICE

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------

def load_image(image_path: str) -> Image.Image:
    """Load image from disk."""
    return Image.open(image_path).convert("RGB")


def crop_from_bbox(image: Image.Image, bbox):
    """Crop PIL image using bbox [x1, y1, x2, y2]."""
    return image.crop(bbox)


def mask_to_numpy(mask_tensor):
    """Convert mask tensor to numpy binary mask."""
    if mask_tensor is None:
        return None
    return mask_tensor.squeeze().cpu().numpy().astype(np.uint8)


# --------------------------------------------------
# Inference
# --------------------------------------------------

class GarmentDetector:
    def __init__(self, checkpoint_path: str):
        self.model = load_model(checkpoint_path, DEVICE)

    @torch.no_grad()
    def predict(self, img: Image.Image) -> List[Dict[str, Any]]:
        self.model.model.model.eval()
        self.model.optimize_for_inference()

        # 1. Run prediction
        outputs_list = self.model.predict([img.convert("RGB")], confidence_threshold=0.2)

        if not outputs_list or outputs_list[0] is None:
            return []

        detections_obj = outputs_list[0]
        results = []

        # 2. Extract data from the Supervision-style Detections object
        # The logs show these exist: .xyxy, .confidence, .class_id, .mask
        for i in range(len(detections_obj.xyxy)):
            bbox = detections_obj.xyxy[i].tolist()  # [x1, y1, x2, y2]
            score = float(detections_obj.confidence[i])
            label_id = int(detections_obj.class_id[i])

            # 3. Handle Mask safely
            mask = None
            if detections_obj.mask is not None:
                mask = detections_obj.mask[i]

            # 4. Crop the garment
            cropped_img = crop_from_bbox(img, bbox)

            results.append({
                "category_id": label_id,
                "category_name": CLASS_NAMES[label_id] if label_id < len(CLASS_NAMES) else f"ID_{label_id}",
                "confidence": score,
                "bbox": bbox,
                "mask": mask,
                "image": cropped_img,
            })

        return results

# --------------------------------------------------
# Example usage
# --------------------------------------------------

if __name__ == "__main__":
    detector = GarmentDetector("output/checkpoint_best_regular.pth")

    # image = load_image("../../examples/ex2.jpeg")
    for img_path in Path("../../examples").glob("*.jpeg"):
        print(f"Processing {img_path}")
        image = load_image(img_path)
        results = detector.predict(image)

        for item in results:
            print(item["category_name"], item["confidence"])