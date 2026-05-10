"""
sample_for_manual_labeling.py
Extracts a diverse, representative subset of crops from the Fashionpedia dataset.
Ensures maximum diversity by restricting to 1 crop per image per category.
"""

import os
import json
import argparse
from collections import defaultdict
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def xywh_to_xyxy(bbox: list[float]) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    return (max(0, int(x)), max(0, int(y)), max(1, int(x + w)), max(1, int(y + h)))


def sample_diverse_crops(img_dir: str, ann_file: str, out_dir: str, samples_per_cat: int = 150):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(ann_file, 'r') as f:
        coco = json.load(f)

    id2file = {img["id"]: img["file_name"] for img in coco["images"]}

    cat_to_seen_images = defaultdict(set)
    cat_to_annotations = defaultdict(list)

    print("Filtering annotations for diversity...")
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        img_id = ann["image_id"]

        if len(cat_to_annotations[cat_id]) >= samples_per_cat:
            continue

        if img_id not in cat_to_seen_images[cat_id]:
            cat_to_seen_images[cat_id].add(img_id)
            cat_to_annotations[cat_id].append(ann)

    metadata = {}
    skipped = 0
    total_to_process = sum(len(anns) for anns in cat_to_annotations.values())

    with tqdm(total=total_to_process, desc="Extracting Crops") as pbar:
        for cat_id, anns in cat_to_annotations.items():
            for ann in anns:
                img_path = os.path.join(img_dir, id2file[ann["image_id"]])
                try:
                    img = Image.open(img_path).convert("RGB")
                    x1, y1, x2, y2 = xywh_to_xyxy(ann["bbox"])

                    if (x2 - x1) < 32 or (y2 - y1) < 32:
                        skipped += 1
                        pbar.update(1)
                        continue

                    crop = img.crop((x1, y1, x2, y2))
                    crop_filename = f"{ann['id']}.jpg"
                    crop.save(out_path / crop_filename)

                    metadata[crop_filename] = {
                        "ann_id": ann["id"],
                        "image_id": ann["image_id"],
                        "category_id": cat_id
                    }
                except Exception:
                    skipped += 1

                pbar.update(1)

    with open(out_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved {len(metadata)} diverse crops. Skipped {skipped}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",  default="../../data/fashionpedia_coco/train")
    parser.add_argument("--ann_file", default="../../data/fashionpedia_coco/train/_annotations.coco.json")
    parser.add_argument("--out_dir", default="manual_labeling_data")
    parser.add_argument("--samples", type=int, default=150)
    args = parser.parse_args()
    sample_diverse_crops(args.img_dir, args.ann_file, args.out_dir, args.samples)