"""
model.py

YOLOS Fashionpedia-based clothing detection utilities.
"""
import os

import torch
from pathlib import Path
from transformers import YolosForObjectDetection, YolosImageProcessor

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
HUB_MODEL = "valentinafeve/yolos-fashionpedia"
HUB_PROCESSOR = "hustvl/yolos-small"

# Local Path
LOCAL_PATH = Path("models/yolos-fashionpedia-local")

# The 46 Fashionpedia categories
CLASS_NAMES = [
    'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest',
    'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat',
    'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer',
    'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar',
    'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique',
    'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel'
]

NUM_CLASSES = len(CLASS_NAMES)


# --------------------------------------------------
# Model Factory
# --------------------------------------------------

def load_model(model_name: str = None, device: str = DEVICE):
    """
    Load the YOLOS model and image processor from local disk if available.
    """
    # Determine the source (Local vs Hub)
    # If a specific path is passed to model_name, use that.
    # Otherwise, check if our default LOCAL_PATH exists.
    if model_name and os.path.isdir(model_name):
        source = model_name
        processor_source = model_name
    elif LOCAL_PATH.exists():
        source = str(LOCAL_PATH)
        processor_source = str(LOCAL_PATH)
        print(f"Loading from local disk: {source}")
    else:
        source = HUB_MODEL
        processor_source = HUB_PROCESSOR
        print(f"Local files not found. Loading from Hugging Face Hub: {source}")

    model = YolosForObjectDetection.from_pretrained(source).to(device)
    processor = YolosImageProcessor.from_pretrained(processor_source)

    return model, processor


if __name__ == "__main__":
    # Test loading
    model, processor = load_model()
    print("Model loaded successfully.")