import argparse
import os
import json
from tqdm import tqdm
from datasets import load_dataset


def prepare_fashionpedia_complete(output_dir):
    # 1. Load Dataset
    print("Loading Fashionpedia from Hugging Face...")
    full_ds = load_dataset("detection-datasets/fashionpedia", split="train")

    # 2. Create 70/20/10 Splits
    print("Creating splits (70% train, 20% valid, 10% test)...")
    # First split off the test set (10%)
    train_valid_test = full_ds.train_test_split(test_size=0.1, seed=42)
    test_ds = train_valid_test["test"]

    # Then split the remaining 90% into train (70% total) and valid (20% total)
    # 20/90 = 0.2222...
    train_valid = train_valid_test["train"].train_test_split(test_size=0.2222, seed=42)

    dataset_splits = {
        "train": train_valid["train"],
        "valid": train_valid["test"],
        "test": test_ds
    }

    # 3. Pre-calculate global categories to ensure consistency across all JSONs
    print("Extracting global categories...")
    unique_cats = set()
    for sample in full_ds:
        unique_cats.update(sample['objects']['category'])

    global_categories = [
        {"id": int(cat_id), "name": f"class_{cat_id}", "supercategory": "fashion"}
        for cat_id in sorted(list(unique_cats))
    ]

    # 4. Process each split
    for split, ds in dataset_splits.items():
        split_path = os.path.join(output_dir, split)
        os.makedirs(split_path, exist_ok=True)

        coco_data = {
            "images": [],
            "annotations": [],
            "categories": global_categories
        }

        print(f"Processing {split} split ({len(ds)} images)...")
        for idx, sample in enumerate(tqdm(ds)):
            file_name = f"{idx}.jpg"
            img_path = os.path.join(split_path, file_name)

            # Save image to disk
            sample['image'].save(img_path)

            # Add image record to COCO
            coco_data["images"].append({
                "id": idx,
                "file_name": file_name,
                "width": sample['image'].size[0],
                "height": sample['image'].size[1]
            })

            # Process annotations
            obj_data = sample['objects']
            segs = obj_data.get('segmentation')

            for i in range(len(obj_data['category'])):
                bbox = obj_data['bbox'][i]

                # Assuming HF format [ymin, xmin, ymax, xmax] maps to COCO [xmin, ymin, w, h]
                coco_bbox = [
                    float(bbox[1]),  # x_min
                    float(bbox[0]),  # y_min
                    float(bbox[3] - bbox[1]),  # width
                    float(bbox[2] - bbox[0])  # height
                ]

                seg = segs[i] if segs and i < len(segs) else []

                coco_data["annotations"].append({
                    "id": len(coco_data["annotations"]) + 1,  # COCO uses 1-indexed annotations
                    "image_id": idx,
                    "category_id": int(obj_data['category'][i]),
                    "bbox": coco_bbox,
                    "area": float(coco_bbox[2] * coco_bbox[3]),
                    "segmentation": seg,
                    "iscrowd": 0
                })

        # Save JSON
        json_path = os.path.join(split_path, "_annotations.coco.json")
        with open(json_path, "w") as f:
            json.dump(coco_data, f)

        print(f"Saved {split} annotations to {json_path}")


# Execute
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the Fashionpedia dataset in COCO format")
    parser.add_argument(
        "--output_dir",
        default="../../data/fashionpedia_coco",
        help="Folder containing split images"
    )
    args = parser.parse_args()

    prepare_fashionpedia_complete(output_dir=args.output_dir)