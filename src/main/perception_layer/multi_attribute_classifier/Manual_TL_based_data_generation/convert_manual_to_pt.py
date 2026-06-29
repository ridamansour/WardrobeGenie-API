"""
convert_manual_to_pt.py
Finalized script to convert manual labels into .pt files.
Ensures perfect compatibility with the AttributeDataset class.
"""

import os
import json
import torch
import argparse
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Constants for label mapping
FIT_MAP = {"Slim": 0, "Regular": 1, "Oversized": 2}
STYLE_MAP = {"Formal": 0, "Casual": 1, "Sport": 2, "Streetwear": 3}
WEATHER_MAP = {"Winter (Freezing)": 1.0, "Transitional (Mild)": 0.5, "Summer (Hot)": 0.0}
FORMALITY_MAP = {
    "Black Tie": 1.0,
    "Business": 0.8,
    "Smart Casual": 0.6,
    "Everyday Casual": 0.3,
    "Gym/Workout": 0.0
}

def to_one_hot(index: int, num_classes: int) -> torch.Tensor:
    tensor = torch.zeros(num_classes, dtype=torch.float32)
    tensor[index] = 1.0
    return tensor

def convert_and_inject(manual_dir: str, train_dir: str):
    manual_path = Path(manual_dir)
    train_path = Path(train_dir)

    # Load metadata (to get category_id and ann_id) and manual labels
    with open(manual_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    with open(manual_path / "manual_labels.json", "r") as f:
        labels = json.load(f)

    # Identical transform used in pseudo-labeling
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    overwrites = 0
    new_files = 0

    for img_filename, label_vals in tqdm(labels.items(), desc="Injecting Manual Labels"):
        img_path = manual_path / img_filename
        if not img_path.exists():
            continue

        try:
            # 1. Load and Transform Image
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)

            # 2. Map labels to exact types expected by AttributeDataset
            meta = metadata[img_filename]

            # One-hot tensors for KL-Divergence loss
            fit_tensor = to_one_hot(FIT_MAP[label_vals["fit"]], len(FIT_MAP))
            style_tensor = to_one_hot(STYLE_MAP[label_vals["style"]], len(STYLE_MAP))

            # Scalar floats for MSE regression
            weather_val = float(WEATHER_MAP[label_vals["weather"]])
            formality_val = float(FORMALITY_MAP[label_vals["formality"]])

            # 3. Construct dictionary with EXACT keys from AttributeDataset
            sample = {
                "image": img_tensor,
                "category_id": int(meta["category_id"]),
                "fit": fit_tensor,
                "style": style_tensor,
                "weather_warmth": weather_val,
                "formality_score": formality_val,
            }

            # 4. OVERWRITE logic: save using the ann_id as the filename
            target_file = train_path / f"{meta['ann_id']}.pt"

            if target_file.exists():
                overwrites += 1
            else:
                new_files += 1

            torch.save(sample, target_file)

        except Exception as e:
            print(f"Error processing {img_filename}: {e}")

    print(f"\nSuccess!")
    print(f"-> Overwrote {overwrites} pseudo-labeled files with ground truth.")
    print(f"-> Added {new_files} new manual labels.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual_dir", default="manual_labeling_data")
    parser.add_argument("--train_dir", required=True, help="The 'attribute_dataset/train' folder")
    args = parser.parse_args()

    convert_and_inject(args.manual_dir, args.train_dir)