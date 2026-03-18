"""
inference.py

Run garment detection using YOLOS Fashionpedia.

Outputs:
- bbox [x1, y1, x2, y2]
- category_id
- category_name
- confidence
- cropped garment image
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any
from pathlib import Path

# Import from our model file
from perception_layer.clothing_detection_yolo_backup.model import load_model, CLASS_NAMES, DEVICE


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------

def box_cxcywh_to_xyxy(x):
    """Convert center-based [cx, cy, w, h] to corner-based [x1, y1, x2, y2]."""
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    """Scale normalized bboxes to actual image pixel dimensions."""
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# --------------------------------------------------
# Inference Class
# --------------------------------------------------

class GarmentDetector:
    def __init__(self, model_path: str = None):
        # We use the model_path if provided, else the default from model.py
        self.model, self.processor = load_model(device=DEVICE)
        self.model.eval()

    @torch.no_grad()
    def predict(self, img: Image.Image, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Run inference and return a list of garment detections.
        """
        # 1. Preprocess
        inputs = self.processor(images=img, return_tensors="pt").to(DEVICE)

        # 2. Forward Pass
        outputs = self.model(**inputs)

        # 3. Post-process logits (softmax to get probabilities)
        # We ignore the last class as it is the "no-object" background class
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold

        # 4. Rescale boxes
        bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), img.size)

        probas_kept = probas[keep]
        results = []

        # 5. Build result objects
        for i, (prob, bbox) in enumerate(zip(probas_kept, bboxes_scaled)):
            bbox_list = bbox.tolist()  # [x1, y1, x2, y2]
            label_id = prob.argmax().item()
            score = prob.max().item()

            # Crop the garment from the original PIL image
            cropped_img = img.crop(bbox_list)
            if CLASS_NAMES[label_id] not in ["sleeve", "pocket", "neckline", 'buckle', 'zipper']:
                results.append({
                    "category_id": label_id,
                    "category_name": CLASS_NAMES[label_id],
                    "confidence": score,
                    "bbox": bbox_list,
                    "mask": None,  # YOLOS is Detection-only, no segmentation mask
                    "image": cropped_img,
                })

        return results


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import List, Dict, Any

# A professional color palette for your presentation
COLORS = [
    "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
    "#ec4899", "#06b6d4", "#84cc16", "#f43f5e"
]


def plot_wardrobe_detections(image: Image.Image, detections: List[Dict[str, Any]], save_path: str = None):
    """
    Plots bounding boxes, labels, and confidence scores onto the image.

    Args:
        image: The original PIL Image.
        detections: The list of dicts returned by GarmentDetector.predict().
        save_path: Optional path to save the resulting plot as an image file.
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    for i, det in enumerate(detections):
        # Extract data from the detection dictionary
        x1, y1, x2, y2 = det["bbox"]
        label = det["category_name"]
        score = det["confidence"]

        # Cycle through our color palette
        color = COLORS[i % len(COLORS)]

        # 1. Create the Bounding Box rectangle
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # 2. Create the Label String (e.g., "Sweater 0.94")
        label_text = f"{label} {score:.2f}"

        # 3. Add the Label Background & Text
        # We place the text slightly above the top-left corner
        ax.text(
            x1, y1 - 5, label_text,
            color='white', fontsize=10, fontweight='bold',
            bbox=dict(facecolor=color, edgecolor='none', alpha=0.8, pad=2)
        )

    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {save_path}")

    plt.show()

# --------------------------------------------------
# Example usage
# --------------------------------------------------

if __name__ == "__main__":
    # Initialize detector
    detector = GarmentDetector()

    # Define path to examples
    example_path = Path("../../examples")

    for img_path in example_path.glob("*.jpeg"):
        print(f"\nProcessing {img_path.name}...")
        image = Image.open(img_path).convert("RGB")

        detections = detector.predict(image, threshold=0.7)

        for item in detections:
            print(f" - Detected {item['category_name']} with {item['confidence']:.2f} confidence.")
            # item["image"].show() # Uncomment to see individual crops