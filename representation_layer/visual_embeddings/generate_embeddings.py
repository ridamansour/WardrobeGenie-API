import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from model import TeacherEncoder
from transformers import CLIPImageProcessor


def generate_dataset_embeddings(data_root, split="train"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup paths
    split_dir = os.path.join(data_root, split)
    ann_path = os.path.join(split_dir, "_annotations.coco.json")
    output_dir = os.path.join(split_dir, "teacher_embeddings")
    os.makedirs(output_dir, exist_ok=True)

    # Load Teacher
    print(f"Loading Teacher model for {split} split...")
    teacher = TeacherEncoder().to(device).eval()
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    with open(ann_path, 'r') as f:
        data = json.load(f)

    images_map = {img['id']: img['file_name'] for img in data['images']}

    print(f"Processing {len(data['annotations'])} annotations...")
    for ann in tqdm(data['annotations']):
        ann_id = ann['id']
        img_filename = images_map.get(ann['image_id'])
        img_path = os.path.join(split_dir, img_filename)

        save_path = os.path.join(output_dir, f"{ann_id}.pt")
        if os.path.exists(save_path): continue  # Skip if already done

        try:
            full_img = Image.open(img_path).convert("RGB")
            x, y, w, h = ann['bbox']
            cropped_img = full_img.crop((x, y, x + w, y + h))

            # Preprocess and embed
            inputs = processor(images=cropped_img, return_tensors="pt").to(device)
            with torch.no_grad():
                embedding = teacher(inputs['pixel_values'])  # (1, 512)

            # Save to disk as float16 to save space, or float32 for precision
            torch.save(embedding.cpu().squeeze(0), save_path)

        except Exception as e:
            print(f"Error processing ann {ann_id}: {e}")


if __name__ == "__main__":
    # Run for both splits
    base_path = "../../data/fashionpedia_coco"
    generate_dataset_embeddings(base_path, "train")
    generate_dataset_embeddings(base_path, "valid")