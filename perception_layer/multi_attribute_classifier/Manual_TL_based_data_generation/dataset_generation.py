"""
dataset_generation.py
Generates pseudo-labeled training data using the FINE-TUNED CLIP model.
Combines batching for speed with Temperature Scaling (TAU) and Category Priors.
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms

TAU = 0.05

CATEGORY_PRIORS = {
    4: {"weather": (0.7, 1.0), "formality": (0.3, 0.9)},  # jacket
    1: {"weather": (0.0, 0.4), "formality": (0.1, 0.6)},  # t-shirt
    6: {"weather": (0.3, 0.8), "formality": (0.2, 0.7)},  # pants
    7: {"weather": (0.0, 0.5), "formality": (0.1, 0.5)},  # shorts
}

TEMPLATE = "a photo of someone wearing a garment perfect for {}"

WEATHER_PROMPTS = [
    TEMPLATE.format("freezing winter weather"),
    TEMPLATE.format("mild, layered transitional weather"),
    TEMPLATE.format("hot, sunny summer weather"),
]

FORMALITY_PROMPTS = [
    TEMPLATE.format("a formal black tie event"),
    TEMPLATE.format("a professional business office"),
    TEMPLATE.format("a smart casual dinner"),
    TEMPLATE.format("everyday casual wear"),
    TEMPLATE.format("working out at the gym"),
]

FIT_PROMPTS = [
    "a photo of a garment with a tight, slim fit",
    "a photo of a garment with a regular, standard fit",
    "a photo of a garment with an oversized, baggy fit",
]

STYLE_PROMPTS = [
    "a photo of a formal style garment",
    "a photo of a casual style garment",
    "a photo of a sport style garment",
    "a photo of a streetwear style garment",
]

FORMALITY_WEIGHTS = torch.tensor([1.0, 0.8, 0.6, 0.3, 0.0])
WEATHER_WEIGHTS = torch.tensor([1.0, 0.5, 0.0])


class CLIPPseudoLabelerBatched:
    def __init__(self, model_name: str, device: str = None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading CLIP from {model_name} on {self.device}...")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)

        self._weather_txt = self._encode_text(WEATHER_PROMPTS)
        self._formality_txt = self._encode_text(FORMALITY_PROMPTS)
        self._fit_txt = self._encode_text(FIT_PROMPTS)
        self._style_txt = self._encode_text(STYLE_PROMPTS)

    @torch.no_grad()
    def _encode_text(self, prompts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        output = self.model.get_text_features(**inputs)
        return F.normalize(output, dim=-1)

    def _get_logits(self, img_feats: torch.Tensor, txt_feats: torch.Tensor) -> torch.Tensor:
        logits = (img_feats @ txt_feats.T) / TAU
        return logits.softmax(dim=-1)

    def _scalar_from_probs(self, probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.to(probs.device)
        max_prob, idx = probs.max(dim=-1)
        confident = max_prob > 0.6
        expected = (probs * weights).sum(dim=-1)
        hard = weights[idx]
        return torch.where(confident, hard, expected)

    @torch.no_grad()
    def label_batch(self, pil_images: list[Image.Image]) -> dict[str, torch.Tensor]:
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        img_f = self.model.get_image_features(**inputs)
        img_f = F.normalize(img_f, dim=-1)

        w_probs = self._get_logits(img_f, self._weather_txt)
        weather = self._scalar_from_probs(w_probs, WEATHER_WEIGHTS).cpu()

        f_probs = self._get_logits(img_f, self._formality_txt)
        formality = self._scalar_from_probs(f_probs, FORMALITY_WEIGHTS).cpu()

        fit = self._get_logits(img_f, self._fit_txt).cpu()
        style = self._get_logits(img_f, self._style_txt).cpu()

        return {
            "fit": fit,
            "style": style,
            "weather_warmth": weather,
            "formality_score": formality,
        }


def xywh_to_xyxy(bbox: list[float]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return (max(0, int(x)), max(0, int(y)), max(1, int(x + w)), max(1, int(y + h)))


def apply_category_prior(value: float, cat_id: int, key: str) -> float:
    if cat_id not in CATEGORY_PRIORS:
        return value
    low, high = CATEGORY_PRIORS[cat_id][key]
    return low + (high - low) * value


def build_attribute_dataset(img_dir: str, ann_file: str, out_dir: str, batch_size: int, clip_model: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(ann_file) as f:
        coco = json.load(f)

    id2file = {img["id"]: img["file_name"] for img in coco["images"]}
    labeler = CLIPPseudoLabelerBatched(model_name=clip_model)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    batch_crops = []
    batch_meta = []

    def flush_batch():
        if not batch_crops: return
        labels = labeler.label_batch(batch_crops)

        for i, meta in enumerate(batch_meta):
            cat_id = meta["category_id"]

            # Apply logical priors before saving
            weather = apply_category_prior(labels["weather_warmth"][i].item(), cat_id, "weather")
            formality = apply_category_prior(labels["formality_score"][i].item(), cat_id, "formality")

            sample = {
                "image": transform(batch_crops[i]),
                "category_id": cat_id,
                "fit": labels["fit"][i],
                "style": labels["style"][i],
                "weather_warmth": weather,
                "formality_score": formality,
            }
            torch.save(sample, os.path.join(out_dir, f"{meta['ann_id']}.pt"))

        batch_crops.clear()
        batch_meta.clear()

    skipped = 0
    for ann in tqdm(coco["annotations"], desc="Labeling Dataset"):
        img_path = os.path.join(img_dir, id2file[ann["image_id"]])
        try:
            img = Image.open(img_path).convert("RGB")
            x1, y1, x2, y2 = xywh_to_xyxy(ann["bbox"])
            if (x2 - x1) < 32 or (y2 - y1) < 32: raise ValueError
            crop = img.crop((x1, y1, x2, y2))

            batch_crops.append(crop)
            batch_meta.append({"ann_id": ann["id"], "category_id": ann["category_id"]})

            if len(batch_crops) >= batch_size: flush_batch()
        except Exception:
            skipped += 1

    flush_batch()
    print(f"Done. Skipped {skipped} invalid/tiny crops.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--ann_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    # Default to the fine-tuned directory produced by script 3
    parser.add_argument("--clip_model", default="./fine_tuned_clip")
    args = parser.parse_args()

    build_attribute_dataset(args.img_dir, args.ann_file, args.out_dir, args.batch_size, args.clip_model)