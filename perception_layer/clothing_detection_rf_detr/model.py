"""
model.py

RF-DETR Segmentation Nano training & loading utilities
for Fashionpedia-based clothing detection.

This module is part of the perception layer:
- Detect garments
- Provide bbox, mask, category, confidence
"""
import os
from pathlib import Path
import torch
from rfdetr import RFDETRNano

# --------------------------------------------------
# Configuration
# --------------------------------------------------

# CLASS_NAMES = [
#     'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest',
#     'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat',
#     'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer',
#     'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar',
#     'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique',
#     'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel'
# ]

CLASS_NAMES = [
    'shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest',
    'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat',
    'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer',
    'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella'
]
NUM_CLASSES = len(CLASS_NAMES)

DATASET_DIR = Path("../../data/fashionpedia_coco")
OUTPUT_DIR = Path("../../models/rf_detr_fashionopedia")

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

TRAIN_CONFIG = {
    "batch_size": 4,
    "grad_accum_steps": 4,
    "num_workers": 2,
    "epochs": 100,
    "lr": 5e-5,
    "weight_decay": 1e-4,
    "use_ema": True,
    "early_stopping": True,
    "early_stopping_patience": 10,
    "resolution": 512,
    "multi_scale": True,
    "checkpoint_interval": 5,
    "tensorboard": True,
}

# --------------------------------------------------
# Model Factory
# --------------------------------------------------

def create_model(device: str = DEVICE, checkpoint_path: str = None) -> RFDETRNano:
    """
    Create RF-DETR Nano model, optionally loading from a checkpoint.
    """
    if checkpoint_path:
        print(f"Loading weights from: {checkpoint_path}")
        return RFDETRNano(
            num_classes=NUM_CLASSES,
            device=device,
            pretrain_weights=checkpoint_path
        )
    return RFDETRNano(num_classes=NUM_CLASSES, device=device)

# --------------------------------------------------
# Training
# --------------------------------------------------

def train_model(
    dataset_dir: Path = DATASET_DIR,
    output_dir: Path = OUTPUT_DIR,
    device: str = DEVICE,
    resume_checkpoint: str = None):
    """
    Execute the distributed training loop.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model (from scratch or resuming)
    model = create_model(device, checkpoint_path=resume_checkpoint)

    print(f"Starting training on {device}...")
    print(f"Dataset: {dataset_dir}")
    print(f"Output:  {output_dir}")

    # The rfdetr .train() method will handle DDP internally
    # when launched via torchrun.
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        device=device,
        **TRAIN_CONFIG,
    )

    return model


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

def load_model(checkpoint_path: str, device: str = DEVICE) -> RFDETRNano:
    """
    Load a trained checkpoint for inference.
    """
    model = create_model(device, checkpoint_path)

    return model


# --------------------------------------------------
# Example usage
# --------------------------------------------------

if __name__ == "__main__":
    train_model()