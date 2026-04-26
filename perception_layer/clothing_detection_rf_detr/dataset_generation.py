import os
import json
from tqdm import tqdm
from datasets import load_dataset


def prepare_fashionpedia_filtered(output_dir="../../data/fashionpedia_coco"):
    # 1. Load Dataset
    print("Loading Fashionpedia...")
    full_ds = load_dataset("detection-datasets/fashionpedia", split="train")

    # 2. Define target classes (WardrobeGenie Core)
    # We are skipping IDs 27-45 (the 'parts' like pockets, zippers, etc.)
    TARGET_CLASS_NAMES = [
        'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest',
        'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat',
        'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer',
        'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella'
    ]

    # Map old Fashionpedia ID -> New Sequential ID (0 to 26)
    id_map = {original_id: i for i, original_id in enumerate(range(len(TARGET_CLASS_NAMES)))}

    # Create category list for COCO JSON
    global_categories = [
        {"id": i, "name": name, "supercategory": "fashion"}
        for i, name in enumerate(TARGET_CLASS_NAMES)
    ]

    # 3. Create 70/20/10 Splits
    train_valid_test = full_ds.train_test_split(test_size=0.1, seed=42)
    test_ds = train_valid_test["test"]
    train_valid = train_valid_test["train"].train_test_split(test_size=0.2222, seed=42)

    dataset_splits = {"train": train_valid["train"], "valid": train_valid["test"], "test": test_ds}

    # 4. Process Splits
    for split, ds in dataset_splits.items():
        split_path = os.path.join(output_dir, split)
        os.makedirs(split_path, exist_ok=True)

        coco_data = {"images": [], "annotations": [], "categories": global_categories}

        print(f"Processing {split} split...")
        for idx, sample in enumerate(tqdm(ds)):
            obj_data = sample['objects']

            # Check if this image has any of our target categories
            valid_indices = [i for i, cat in enumerate(obj_data['category']) if cat in id_map]

            if not valid_indices:
                continue  # Skip images that only contain 'parts' like zippers

            # Save Image
            file_name = f"{idx}.jpg"
            sample['image'].save(os.path.join(split_path, file_name))
            img_w, img_h = sample['image'].size

            coco_data["images"].append({
                "id": idx, "file_name": file_name, "width": img_w, "height": img_h
            })

            # Process only valid annotations
            for i in valid_indices:
                bbox = obj_data['bbox'][i]

                # Conversion to COCO: [xmin, ymin, width, height]
                # Fashionpedia format is [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

                coco_bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

                coco_data["annotations"].append({
                    "id": len(coco_data["annotations"]) + 1,
                    "image_id": idx,
                    "category_id": id_map[obj_data['category'][i]],  # Use the new sequential ID
                    "bbox": coco_bbox,
                    "area": float(coco_bbox[2] * coco_bbox[3]),
                    "iscrowd": 0
                })

        # Save JSON
        with open(os.path.join(split_path, "_annotations.coco.json"), "w") as f:
            json.dump(coco_data, f)


if __name__ == "__main__":
    prepare_fashionpedia_filtered()