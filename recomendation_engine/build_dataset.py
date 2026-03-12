import os
import argparse
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter
from itertools import combinations

# Import your Perception modules
from perception_layer.clothing_detection_segmintation import inference as det_inference
from perception_layer.multi_attribute_classifire import inference as attr_inference
from perception_layer import color_utils


# Placeholder for your Vectorizer (e.g., Student CLIP)
from representation_layer.visual_embeddings.inference import GarmentEmbedder

def build_co_occurrence_matrix(all_outfit_categories):
    """Builds the category co-occurrence matrix for negative sampling."""
    pair_counts = Counter()
    for cats in all_outfit_categories:
        sorted_cats = sorted(cats)
        for pair in combinations(sorted_cats, 2):
            pair_counts[pair] += 1

    matrix = {}
    max_val = max(pair_counts.values()) if pair_counts else 1
    for pair, count in pair_counts.items():
        matrix[pair] = count / max_val
    return matrix


def main(dataset_path, det_model_path, attr_model_path, output_file):
    print(f"Loading Perception Models...")
    garment_detector = det_inference.GarmentDetector(det_model_path)
    attribute_predictor = attr_inference.AttributePredictor(attr_model_path)
    embedder = GarmentEmbedder() # Initialize your 512-dim vectorizer here

    image_paths = list(Path(dataset_path).glob("*.jpg")) + list(Path(dataset_path).glob("*.jpeg"))
    print(f"Found {len(image_paths)} images in {dataset_path}")

    processed_outfits = []
    wardrobe_pool = []
    all_outfit_categories = []

    for img_path in tqdm(image_paths, desc="Processing Fashionpedia"):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue

        # 1. Detect Garments
        garments = garment_detector.predict(img)
        if len(garments) < 2:
            continue  # Skip images that don't have at least a top/bottom pairing

        outfit_items = []
        outfit_cats = []

        for garment in garments:
            cropped_img = garment["image"].convert("RGB")

            # 2. Extract Attributes
            attributes = attribute_predictor.predict(cropped_img)

            # 3. Extract Dominant Color
            # quantize_colors returns [(hex, pct), ...] - we take the top one
            dominant_colors = color_utils.quantize_colors(cropped_img, k=1)
            primary_hex = dominant_colors[0][0] if dominant_colors else "#FFFFFF"
            primary_pct = dominant_colors[0][1] if dominant_colors else 1.0

            # 4. Extract 512-dim Vector (Required for the Stylist model)
            vector = embedder.embed_crop(cropped_img)

            item_data = {
                "category_id": garment["category_id"],
                "attributes": attributes,
                "color_hex": primary_hex,
                "color_pct": primary_pct,
                "vector": vector
            }

            outfit_items.append(item_data)
            outfit_cats.append(garment["category_id"])
            wardrobe_pool.append(item_data)

        # Calculate Outfit Harmony using your util
        outfit_cropped_imgs = [g["image"].convert("RGB") for g in garments]
        harmony = color_utils.harmony_score_from_images(outfit_cropped_imgs)

        processed_outfits.append({
            "items": outfit_items,
            "harmony_score": harmony
        })
        all_outfit_categories.append(outfit_cats)

    # 5. Build Co-occurrence matrix for negative sampling
    co_occurrence_matrix = build_co_occurrence_matrix(all_outfit_categories)

    # 6. Serialize the dataset
    dataset_bundle = {
        "outfits": processed_outfits,
        "wardrobe_pool": wardrobe_pool,
        "co_occurrence_matrix": co_occurrence_matrix
    }

    torch.save(dataset_bundle, output_file)
    print(f"Successfully saved {len(processed_outfits)} processed outfits to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Fashionpedia into Brain Layer tensors.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to Fashionpedia images")
    parser.add_argument("--output", type=str, default="processed_fashion_data.pt", help="Output file path")

    # Model paths
    parser.add_argument("--det_model", type=str,
                        default="perception_layer/clothing_detection_segmintation/output/checkpoint_best_regular.pth")
    parser.add_argument("--attr_model", type=str,
                        default="perception_layer/multi_attribute_classifire/runs/1b/best_model.pt")

    args = parser.parse_args()

    main(args.dataset_path, args.det_model, args.attr_model, args.output)