import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from representation_layer.visual_embeddings.inference import GarmentEmbedder


def process_fashionpedia(json_path, img_dir, output_file="processed_pool.pt"):
    """
    Parses Fashionpedia, crops items, vectorizes them, and saves a flattened pool.
    """
    embedder = GarmentEmbedder("../representation_layer/visual_embeddings/best_student_model.pth", device="cuda")

    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    anns_by_img = defaultdict(list)
    for ann in data['annotations']:
        anns_by_img[ann['image_id']].append(ann)

    outfits = []
    wardrobe_pool = []

    # FIX: Create a directory to store the physical cropped images
    crops_dir = os.path.join(os.path.dirname(output_file), "crops")
    os.makedirs(crops_dir, exist_ok=True)

    print(f"Processing images in {img_dir}...")
    for img_id, anns in tqdm(anns_by_img.items()):
        if len(anns) < 2: continue  # Ignore single-garment images

        img_info = images[img_id]
        img_path = os.path.join(img_dir, img_info['file_name'])
        try:
            full_img = Image.open(img_path).convert("RGB")
        except:
            continue

        current_outfit = {"vecs": [], "cats": [], "crop_paths": []}

        for ann in anns:
            # 1. Get cropped images of given bboxes
            x, y, w, h = ann['bbox']
            crop = full_img.crop((x, y, x + w, y + h))

            crop_filename = f"{img_id}_{ann['id']}.jpg"
            crop_path = os.path.join(crops_dir, crop_filename)
            crop.save(crop_path)

            # 2. Vectorize using GarmentEmbedder
            vec = embedder.embed_crop(crop)

            # FIX: Store the crop_path string instead of the PIL Image object
            item = {"vec": vec, "cat": ann['category_id'], "crop_path": crop_path}
            current_outfit["vecs"].append(vec)
            current_outfit["cats"].append(ann['category_id'])
            current_outfit["crop_paths"].append(crop_path)
            wardrobe_pool.append(item)

        outfits.append(current_outfit)

    # Serialize for training (now memory-safe)
    torch.save({"outfits": outfits, "pool": wardrobe_pool}, output_file)
    print(f"Dataset built: {len(outfits)} outfits, {len(wardrobe_pool)} total items.")


if __name__ == "__main__":
    process_fashionpedia(
        json_path="../data/fashionpedia_coco/train/_annotations.coco.json",
        img_dir="../data/fashionpedia_coco/train",
        output_file="../data/fashionpedia_coco/processed_pool.pt"
    )