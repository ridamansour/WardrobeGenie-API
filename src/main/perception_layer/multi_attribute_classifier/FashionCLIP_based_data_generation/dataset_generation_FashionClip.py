"""
1.B — Multi-Attribute Classifier: Dataset Generation FashionCLIP
=====================================================
Generates pseudo-labeled training data for the EfficientNet-B0 multi-head
attribute classifier by running FashionCLIP over Fashionpedia crops.

Pipeline:
    Fashionpedia COCO annotations
        → crop garments
        → FashionCLIP pseudo-label
        → save .pt dataset samples

Important:
    FashionCLIP is used ONLY for pseudo-labeling.
    The saved image tensor uses ImageNet normalization because the student model
    is EfficientNet-B0.

Usage:
    python 1b_dataset_generation.py \
        --img_dir  fashionpedia_coco/train \
        --ann_file fashionpedia_coco/train/_annotations.coco.json \
        --out_dir  attribute_dataset/train \
        --batch_size 64

    python 1b_dataset_generation.py \
        --img_dir  fashionpedia_coco/valid \
        --ann_file fashionpedia_coco/valid/_annotations.coco.json \
        --out_dir  attribute_dataset/valid \
        --batch_size 64
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

FASHIONCLIP_MODEL = "patrickjohncyh/fashion-clip"

# Lower = sharper probabilities.
TAU = 0.01

# Fashionpedia filtered classes, remapped by your COCO generator to 0..26.
CATEGORY_ID_TO_NAME = {
    0: "shirt, blouse", 1: "top, t-shirt, sweatshirt", 2: "sweater", 3: "cardigan", 4: "jacket", 5: "vest", 6: "pants",
    7: "shorts", 8: "skirt", 9: "coat", 10: "dress", 11: "jumpsuit", 12: "cape", 13: "glasses", 14: "hat",
    15: "headband, head covering, hair accessory", 16: "tie", 17: "glove", 18: "watch", 19: "belt", 20: "leg warmer",
    21: "tights, stockings", 22: "sock", 23: "shoe", 24: "bag, wallet", 25: "scarf", 26: "umbrella",
}

# EfficientNet-B0 expects ImageNet normalization.
STUDENT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ──────────────────────────────────────────────────────────────────────────────
# Prompt Banks
# ──────────────────────────────────────────────────────────────────────────────

WEATHER_PROMPTS = [
    "a fashion garment suitable for cold winter weather, heavy insulated clothing, thick coat, scarf or gloves",
    "a fashion garment suitable for mild transitional weather, layered clothing, light jacket or sweater",
    "a fashion garment suitable for hot summer weather, light breathable clothing, short sleeves or thin fabrics",
]

# Higher weather_warmth = warmer clothing / colder-weather suitability.
WEATHER_WEIGHTS = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)

FORMALITY_PROMPTS = [
    "black tie formal evening wear, tuxedo or elegant evening gown",
    "business professional suit or formal office attire",
    "smart business casual outfit, neat and polished",
    "casual everyday clothing, relaxed and comfortable",
    "athletic sportswear or gym training outfit",
]

# Higher formality_score = more formal.
FORMALITY_WEIGHTS = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], dtype=torch.float32)

FIT_PROMPTS = [
    "tight slim fit clothing closely fitted to the body",
    "regular fit clothing with standard comfortable cut",
    "oversized baggy loose clothing with wide silhouette",
]

STYLE_PROMPTS = [
    "formal elegant fashion style",
    "casual everyday fashion style",
    "athletic sport activewear style",
    "streetwear urban fashion style",
]


# Optional logical priors for scalar values.
# These are nudges, not hard labels.
CATEGORY_PRIORS = {
    1: {"weather": (0.0, 0.5), "formality": (0.0, 0.6)},   # top, t-shirt, sweatshirt
    2: {"weather": (0.5, 0.9), "formality": (0.1, 0.6)},   # sweater
    3: {"weather": (0.4, 0.8), "formality": (0.2, 0.7)},   # cardigan
    4: {"weather": (0.7, 1.0), "formality": (0.2, 0.8)},   # jacket
    6: {"weather": (0.2, 0.8), "formality": (0.1, 0.8)},   # pants
    7: {"weather": (0.0, 0.4), "formality": (0.0, 0.5)},   # shorts
    9: {"weather": (0.8, 1.0), "formality": (0.3, 0.9)},   # coat
    20: {"weather": (0.5, 1.0), "formality": (0.0, 0.4)},  # leg warmer
    25: {"weather": (0.3, 0.9), "formality": (0.1, 0.7)},  # scarf
    26: {"weather": (0.4, 1.0), "formality": (0.0, 0.6)},  # umbrella
}


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def apply_category_prior(value: float, category_id: int, key: str) -> float:
    prior = CATEGORY_PRIORS.get(int(category_id))

    if prior is None or key not in prior:
        return float(max(0.0, min(1.0, value)))

    low, high = prior[key]
    adjusted = low + (high - low) * value

    return float(max(0.0, min(1.0, adjusted)))


def xywh_to_xyxy(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    COCO [x, y, w, h] → clipped [x1, y1, x2, y2].
    """
    x, y, w, h = bbox

    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(image_width, int(round(x + w)))
    y2 = min(image_height, int(round(y + h)))

    return x1, y1, x2, y2


# ──────────────────────────────────────────────────────────────────────────────
# FashionCLIP Pseudo-Labeler
# ──────────────────────────────────────────────────────────────────────────────

class CLIPPseudoLabeler:
    """
    Wraps FashionCLIP to produce soft pseudo-labels for clothing attributes.
    All text prompts are encoded once and cached for speed.
    """

    def __init__(
        self,
        model_name: str = FASHIONCLIP_MODEL,
        device: Optional[str] = None,
    ):
        self.device = device or get_best_device()

        print(f"[CLIPPseudoLabeler] Loading {model_name} on {self.device}...")

        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)

        self._weather_txt = self._encode_text(WEATHER_PROMPTS)
        self._formality_txt = self._encode_text(FORMALITY_PROMPTS)
        self._fit_txt = self._encode_text(FIT_PROMPTS)
        self._style_txt = self._encode_text(STYLE_PROMPTS)

    @torch.no_grad()
    def _encode_text(self, prompts: list[str]) -> torch.Tensor:
        inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, dim=-1)

    @staticmethod
    def _get_probs(
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        logits = (image_features @ text_features.T) / TAU
        return logits.softmax(dim=-1)

    @staticmethod
    def _weighted_scalar(
            probs: torch.Tensor,
            weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes a pure continuous expected value. Completely removes the
        hard threshold gate to allow smooth, organic distributions.
        """
        weights = weights.to(probs.device)
        return (probs * weights).sum(dim=-1)

    @torch.no_grad()
    def label_batch(self, pil_images: list[Image.Image]) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        pil_images:
            List of cropped garment PIL images.

        Returns
        -------
        fit:
            (B, 3) soft probs [slim, regular, oversized]

        style:
            (B, 4) soft probs [formal, casual, athletic, streetwear]

        weather_warmth:
            (B,) scalar.
            Higher = warmer clothing / colder-weather suitability.

        formality_score:
            (B,) scalar.
            Higher = more formal.
        """
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt",
        ).to(self.device)

        img_f = self.model.get_image_features(**inputs)
        img_f = F.normalize(img_f, dim=-1)

        weather_probs = self._get_probs(img_f, self._weather_txt)
        formality_probs = self._get_probs(img_f, self._formality_txt)
        fit_probs = self._get_probs(img_f, self._fit_txt)
        style_probs = self._get_probs(img_f, self._style_txt)

        weather_warmth = self._weighted_scalar(weather_probs, WEATHER_WEIGHTS)
        formality_score = self._weighted_scalar(formality_probs, FORMALITY_WEIGHTS)

        return {
            "fit": fit_probs.cpu(),
            "style": style_probs.cpu(),
            "weather_warmth": weather_warmth.cpu(),
            "formality_score": formality_score.cpu(),
            "weather_probs": weather_probs.cpu(),
            "formality_probs": formality_probs.cpu(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_attribute_dataset(
    img_dir: str,
    ann_file: str,
    out_dir: str,
    batch_size: int = 64,
    clip_model: str = FASHIONCLIP_MODEL,
    min_crop_px: int = 32,
    overwrite: bool = False,
    use_category_priors: bool = True,
) -> None:
    """
    Reads a COCO-format annotation file, crops every annotated garment,
    pseudo-labels it with FashionCLIP, and saves individual sample .pt files.
    """
    img_dir_path = Path(img_dir)
    ann_file_path = Path(ann_file)
    out_dir_path = Path(out_dir)

    out_dir_path.mkdir(parents=True, exist_ok=True)

    with open(ann_file_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id2file = {
        int(img["id"]): img["file_name"]
        for img in coco["images"]
    }

    annotations = coco["annotations"]

    labeler = CLIPPseudoLabeler(model_name=clip_model)

    print(f"[build] {len(annotations)} annotations → batch_size={batch_size}")

    batch_crops: list[Image.Image] = []
    batch_meta: list[dict] = []

    skipped = 0
    saved = 0

    def flush_batch() -> None:
        nonlocal saved

        if not batch_crops:
            return

        labels = labeler.label_batch(batch_crops)

        for i, meta in enumerate(batch_meta):
            out_path = out_dir_path / f"{meta['ann_id']}.pt"

            if out_path.exists() and not overwrite:
                continue

            category_id = int(meta["category_id"])

            weather = float(labels["weather_warmth"][i].item())
            formality = float(labels["formality_score"][i].item())

            weather_raw = weather
            formality_raw = formality

            if use_category_priors:
                weather = apply_category_prior(weather, category_id, "weather")
                formality = apply_category_prior(formality, category_id, "formality")

            crop_tensor = STUDENT_TRANSFORM(batch_crops[i])

            sample = {
                "image": crop_tensor,
                "category_id": category_id,
                "category_name": CATEGORY_ID_TO_NAME.get(category_id, str(category_id)),
                "fit": labels["fit"][i].float(),
                "style": labels["style"][i].float(),
                "weather_warmth": weather,
                "formality_score": formality,
                "metadata": {
                    "ann_id": int(meta["ann_id"]),
                    "image_id": int(meta["image_id"]),
                    "file_name": meta["file_name"],
                    "bbox_xywh": [float(v) for v in meta["bbox"]],
                    "weather_warmth_raw": weather_raw,
                    "formality_score_raw": formality_raw,
                    "clip_model": clip_model,
                    "use_category_priors": use_category_priors,
                },
            }

            torch.save(sample, out_path)
            saved += 1

        batch_crops.clear()
        batch_meta.clear()

    for ann in tqdm(annotations, desc="Labeling crops"):
        try:
            image_id = int(ann["image_id"])

            if image_id not in id2file:
                skipped += 1
                continue

            file_name = id2file[image_id]
            img_path = img_dir_path / file_name

            if not img_path.exists():
                skipped += 1
                continue

            img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
            img_w, img_h = img.size

            x1, y1, x2, y2 = xywh_to_xyxy(
                bbox=ann["bbox"],
                image_width=img_w,
                image_height=img_h,
            )

            if (x2 - x1) < min_crop_px or (y2 - y1) < min_crop_px:
                skipped += 1
                continue

            crop = img.crop((x1, y1, x2, y2))

            batch_crops.append(crop)
            batch_meta.append({
                "ann_id": int(ann["id"]),
                "image_id": image_id,
                "file_name": file_name,
                "category_id": int(ann["category_id"]),
                "bbox": ann["bbox"],
            })

            if len(batch_crops) >= batch_size:
                flush_batch()

        except Exception:
            skipped += 1

    flush_batch()

    print(f"[build] Done. Saved {saved} samples | Skipped {skipped}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate FashionCLIP pseudo-labels for WardrobeGenie 1.B"
    )

    parser.add_argument("--img_dir", required=True, help="Folder containing split images")
    parser.add_argument("--ann_file", required=True, help="COCO annotation JSON for the split")
    parser.add_argument("--out_dir", required=True, help="Where to save .pt sample files")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_model", default=FASHIONCLIP_MODEL)
    parser.add_argument("--min_crop_px", type=int, default=32)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files",
    )

    parser.add_argument(
        "--no_category_priors",
        action="store_true",
        help="Disable logical category priors for weather/formality scalar labels",
    )

    args = parser.parse_args()

    build_attribute_dataset(
        img_dir=args.img_dir,
        ann_file=args.ann_file,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        clip_model=args.clip_model,
        min_crop_px=args.min_crop_px,
        overwrite=args.overwrite,
        use_category_priors=not args.no_category_priors,
    )