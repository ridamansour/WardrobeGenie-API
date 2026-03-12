import argparse
import os
import json
from tqdm import tqdm
from datasets import load_dataset

def prepare_fashionpedia_complete(output_dir="../../data/fashionpedia_coco"):
    # 1. Load Dataset
    print("Loading Fashionpedia...")
    full_ds = load_dataset("detection-datasets/fashionpedia", split="train")

    # 2. Create 70/20/10 Splits
    # First split off the test set (10%)
    train_valid_test = full_ds.train_test_split(test_size=0.1, seed=42)
    test_ds = train_valid_test["test"]

    # Then split the remaining 90% into train (70% total) and valid (20% total)
    # 0.222 roughly equals 20% of the original 100%
    train_valid = train_valid_test["train"].train_test_split(test_size=0.222, seed=42)

    dataset_splits = {
        "train": train_valid["train"],
        "valid": train_valid["test"],
        "test": test_ds
    }

    for split, ds in dataset_splits.items():
        split_path = os.path.join(output_dir, split)
        os.makedirs(split_path, exist_ok=True)

        coco_data = {"images": [], "annotations": [], "categories": []}
        unique_cats = set()

        print(f"Processing {split} split ({len(ds)} images)...")
        for idx, sample in enumerate(tqdm(ds)):
            file_name = f"{idx}.jpg"
            img_path = os.path.join(split_path, file_name)
            sample['image'].save(img_path)

            coco_data["images"].append({
                "id": idx, "file_name": file_name,
                "width": sample['image'].size[0], "height": sample['image'].size[1]
            })

            obj_data = sample['objects']
            segs = obj_data.get('segmentation')
            for i in range(len(obj_data['category'])):
                bbox = obj_data['bbox'][i]
                coco_bbox = [float(bbox[1]), float(bbox[0]), float(bbox[3]-bbox[1]), float(bbox[2]-bbox[0])]
                seg = segs[i] if segs and i < len(segs) else []

                coco_data["annotations"].append({
                    "id": len(coco_data["annotations"]),
                    "image_id": idx,
                    "category_id": int(obj_data['category'][i]),
                    "bbox": coco_bbox,
                    "area": float(coco_bbox[2] * coco_bbox[3]),
                    "segmentation": seg,
                    "iscrowd": 0
                })
                unique_cats.add(obj_data['category'][i])

        for cat_id in sorted(list(unique_cats)):
            coco_data["categories"].append({
                "id": int(cat_id), "name": f"class_{cat_id}", "supercategory": "fashion"
            })

        with open(os.path.join(split_path, "_annotations.coco.json"), "w") as f:
            json.dump(coco_data, f)

# Execute
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the Fasionopedia dataset in coco format")
    parser.add_argument("--output_dir",    required=False, help="Folder containing split images")
    args = parser.parse_args()

    prepare_fashionpedia_complete(
        output_dir= args.output_dir
    )
prepare_fashionpedia_complete()