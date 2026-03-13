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
    embedder = GarmentEmbedder()

    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    anns_by_img = defaultdict(list)
    for ann in data['annotations']:
        anns_by_img[ann['image_id']].append(ann)

    outfits = []
    wardrobe_pool = []

    print(f"Processing images in {img_dir}...")
    for img_id, anns in tqdm(anns_by_img.items()):
        if len(anns) < 2: continue  # Ignore single-garment images

        img_info = images[img_id]
        img_path = os.path.join(img_dir, img_info['file_name'])
        try:
            full_img = Image.open(img_path).convert("RGB")
        except:
            continue

        current_outfit = {"vecs": [], "cats": [], "crops": []}

        for ann in anns:
            # 1. Get cropped images of given bboxes
            x, y, w, h = ann['bbox']
            crop = full_img.crop((x, y, x + w, y + h))

            # 2. Vectorize using GarmentEmbedder
            vec = embedder.embed_crop(crop)

            item = {"vec": vec, "cat": ann['category_id'], "img": crop}
            current_outfit["vecs"].append(vec)
            current_outfit["cats"].append(ann['category_id'])
            current_outfit["crops"].append(crop)
            wardrobe_pool.append(item)

        outfits.append(current_outfit)

    # Serialize for training
    torch.save({"outfits": outfits, "pool": wardrobe_pool}, output_file)
    print(f"Dataset built: {len(outfits)} outfits, {len(wardrobe_pool)} total items.")


if __name__ == "__main__":
    process_fashionpedia("../data/fashionpedia_coco/train/_annotations.coco.json", "../data/fashionpedia_coco/train")