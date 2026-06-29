"""
1.B — Multi-Attribute Classifier: Dataset Generation
=====================================================
Generates pseudo-labeled training data for the EfficientNet-B0 multi-head
attribute classifier by running CLIP over Fashionpedia crops.

Pipeline:
    Fashionpedia annotations → crop garments → CLIP pseudo-label → save .pt dataset

Usage:
    python 1b_dataset_generation.py \
        --img_dir  fashionpedia_coco/train \
        --ann_file fashionpedia_coco/train/_annotations.coco.json \
        --out_dir  attribute_dataset/train \
        --batch_size 64

    # Repeat for valid / test splits.
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# ──────────────────────────────────────────────────────────────────────────────
# Prompt Banks
# ──────────────────────────────────────────────────────────────────────────────

WEATHER_PROMPTS = [
    "a person wearing heavy insulated winter clothing, thick coat, scarf and gloves",
    "a person wearing layered clothing suitable for cool weather, light jacket or sweater",
    "a person wearing light breathable summer clothing, short sleeves and thin fabrics",
]

FORMALITY_PROMPTS = [
    "black tie formal evening wear, tuxedo or elegant evening gown",
    "business professional suit or formal office attire",
    "smart business casual outfit, neat and polished",
    "casual everyday clothing, relaxed and comfortable",
    "athletic sportswear or gym training outfit",
]
FORMALITY_WEIGHTS = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])   # maps 5 classes → scalar

FIT_PROMPTS = [
    "tight slim fit clothing closely fitted to the body",
    "regular fit clothing with standard comfortable cut",
    "oversized baggy loose clothing with wide silhouette",
]

STYLE_PROMPTS = [
    "formal elegant fashion style",
    "casual everyday fashion style",
    "sport athletic activewear style",
    "streetwear urban fashion style",
]


# ──────────────────────────────────────────────────────────────────────────────
# CLIP Pseudo-Labeler
# ──────────────────────────────────────────────────────────────────────────────

class CLIPPseudoLabeler:
    """
    Wraps OpenAI CLIP to produce soft pseudo-labels for clothing attributes.
    All text prompts are encoded once and cached for speed.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        # OPTIMIZATION: Use Mac GPU (MPS) if available
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[CLIPPseudoLabeler] Loading {model_name} on {self.device} …")

        # FIX: Do NOT pass use_fast to CLIPModel
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

        # FIX: pass use_fast=False HERE to enable slow mode and stop warnings
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)

        # Pre-encode all text prompts once
        self._weather_txt = self._encode_text(WEATHER_PROMPTS)
        self._formality_txt = self._encode_text(FORMALITY_PROMPTS)
        self._fit_txt = self._encode_text(FIT_PROMPTS)
        self._style_txt = self._encode_text(STYLE_PROMPTS)

    @torch.no_grad()
    def _encode_text(self, prompts: list[str]) -> torch.Tensor:
        # FIX: Removed use_fast from the call to avoid "invalid argument" warning
        inputs = self.processor(text=prompts, return_tensors="pt",
                                padding=True).to(self.device)
        output = self.model.get_text_features(**inputs)

        if isinstance(output, torch.Tensor):
            feats = output
        else:
            feats = getattr(output, "text_embeds",
                            getattr(output, "pooler_output", output.last_hidden_state[:, 0]))
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def label_batch(self, pil_images: list[Image.Image]) -> dict[str, torch.Tensor]:
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        _img_out = self.model.get_image_features(**inputs)

        if isinstance(_img_out, torch.Tensor):
            img_f = _img_out
        else:
            img_f = getattr(_img_out, "image_embeds",
                            getattr(_img_out, "pooler_output", _img_out.last_hidden_state[:, 0]))
        img_f = F.normalize(img_f, dim=-1)

        # ── weather warmth ────────────────────────────────────────────────────
        w_probs = (img_f @ self._weather_txt.T).softmax(dim=-1).cpu()
        warmth = (w_probs * torch.tensor([0.0, 0.5, 1.0])).sum(dim=-1)

        # ── formality ─────────────────────────────────────────────────────────
        f_probs = (img_f @ self._formality_txt.T).softmax(dim=-1).cpu()
        formality = (f_probs * FORMALITY_WEIGHTS).sum(dim=-1)

        # ── fit ───────────────────────────────────────────────────────────────
        fit_probs = (img_f @ self._fit_txt.T).softmax(dim=-1).cpu()

        # ── style ─────────────────────────────────────────────────────────────
        style_probs = (img_f @ self._style_txt.T).softmax(dim=-1).cpu()

        return {
            "fit": fit_probs,
            "style": style_probs,
            "weather_warmth": warmth,
            "formality_score": formality,
        }

    @torch.no_grad()
    def label_batch(self, pil_images: list[Image.Image]) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        pil_images : list of PIL Images (cropped garments)

        Returns (all CPU tensors, batch-first)
        -------
        fit              : (B, 3)   soft probs [slim, regular, oversized]
        style            : (B, 4)   soft probs [formal, casual, athletic, streetwear]
        weather_warmth   : (B,)     scalar 0-cold → 1-warm
        formality_score  : (B,)     scalar 0-pjs  → 1-tux
        """
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        _img_out = self.model.get_image_features(**inputs)
        # Same version-compat fix as _encode_text: extract tensor if needed
        if isinstance(_img_out, torch.Tensor):
            img_f = _img_out
        else:
            img_f = getattr(_img_out, "image_embeds",
                    getattr(_img_out, "pooler_output", _img_out.last_hidden_state[:, 0]))
        img_f  = F.normalize(img_f, dim=-1)                    # (B, 512)

        # ── weather warmth ────────────────────────────────────────────────────
        w_probs = (img_f @ self._weather_txt.T).softmax(dim=-1).cpu()  # (B, 3)
        # cold=0, mild=0.5, warm=1
        warmth  = (w_probs * torch.tensor([0.0, 0.5, 1.0])).sum(dim=-1)

        # ── formality ─────────────────────────────────────────────────────────
        f_probs = (img_f @ self._formality_txt.T).softmax(dim=-1).cpu()  # (B, 5)
        formality = (f_probs * FORMALITY_WEIGHTS).sum(dim=-1)

        # ── fit ───────────────────────────────────────────────────────────────
        fit_probs = (img_f @ self._fit_txt.T).softmax(dim=-1).cpu()      # (B, 3)

        # ── style ─────────────────────────────────────────────────────────────
        style_probs = (img_f @ self._style_txt.T).softmax(dim=-1).cpu()  # (B, 4)

        return {
            "fit":            fit_probs,
            "style":          style_probs,
            "weather_warmth": warmth,
            "formality_score": formality,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Builder
# ──────────────────────────────────────────────────────────────────────────────

def xywh_to_xyxy(bbox: list[float]) -> tuple[int, int, int, int]:
    """COCO [x, y, w, h] → [x1, y1, x2, y2] clipped to >= 0."""
    x, y, w, h = bbox
    return (max(0, int(x)), max(0, int(y)),
            max(1, int(x + w)), max(1, int(y + h)))


def build_attribute_dataset(
    img_dir:    str,
    ann_file:   str,
    out_dir:    str,
    batch_size: int = 32,
    clip_model: str = "openai/clip-vit-base-patch32",
    min_crop_px: int = 32,     # skip tiny crops
) -> None:
    """
    Reads a COCO-format annotation file, crops every annotated garment,
    pseudo-labels it with CLIP, and saves individual sample .pt files.

    Output per sample (out_dir/<ann_id>.pt):
        {
            "image"          : torch.Tensor (3, 224, 224),   # normalized crop
            "category_id"    : int,
            "fit"            : torch.Tensor (3,),
            "style"          : torch.Tensor (4,),
            "weather_warmth" : float,
            "formality_score": float,
        }
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(ann_file) as f:
        coco = json.load(f)

    # Build lookup: image_id → file_name
    id2file = {img["id"]: img["file_name"] for img in coco["images"]}

    labeler = CLIPPseudoLabeler(model_name=clip_model)

    # Standard ImageNet-style normalization (same as CLIP preprocessing)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275,  0.40821073],
            std =[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    annotations = coco["annotations"]
    print(f"[build] {len(annotations)} annotations → processing in batches of {batch_size}")

    # Accumulate batches then flush
    batch_crops:   list[Image.Image] = []
    batch_meta:    list[dict]         = []

    def flush_batch():
        if not batch_crops:
            return
        labels = labeler.label_batch(batch_crops)
        for i, meta in enumerate(batch_meta):
            crop_tensor = transform(batch_crops[i])
            sample = {
                "image":           crop_tensor,
                "category_id":     meta["category_id"],
                "fit":             labels["fit"][i],
                "style":           labels["style"][i],
                "weather_warmth":  labels["weather_warmth"][i].item(),
                "formality_score": labels["formality_score"][i].item(),
            }
            out_path = os.path.join(out_dir, f"{meta['ann_id']}.pt")
            torch.save(sample, out_path)
        batch_crops.clear()
        batch_meta.clear()

    skipped = 0
    for ann in tqdm(annotations, desc="Labeling crops"):
        img_path = os.path.join(img_dir, id2file[ann["image_id"]])
        if not os.path.exists(img_path):
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        x1, y1, x2, y2 = xywh_to_xyxy(ann["bbox"])
        if (x2 - x1) < min_crop_px or (y2 - y1) < min_crop_px:
            skipped += 1
            continue

        crop = img.crop((x1, y1, x2, y2))
        batch_crops.append(crop)
        batch_meta.append({"ann_id": ann["id"], "category_id": ann["category_id"]})

        if len(batch_crops) >= batch_size:
            flush_batch()

    flush_batch()   # remaining
    total = len(annotations) - skipped
    print(f"[build] Done. Saved {total} samples  |  Skipped {skipped}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate attribute pseudo-labels for 1.B")
    parser.add_argument("--img_dir",    required=True, help="Folder containing split images")
    parser.add_argument("--ann_file",   required=True, help="COCO annotation JSON for the split")
    parser.add_argument("--out_dir",    required=True, help="Where to save .pt sample files")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()

    build_attribute_dataset(
        img_dir    = args.img_dir,
        ann_file   = args.ann_file,
        out_dir    = args.out_dir,
        batch_size = args.batch_size,
        clip_model = args.clip_model,
    )
