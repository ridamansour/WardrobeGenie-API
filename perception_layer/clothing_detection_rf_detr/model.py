"""
model.py

RF-DETR Segmentation Nano training & loading utilities
for Fashionpedia-based clothing detection.

This module is part of the perception layer:
- Detect garments
- Provide bbox, mask, category, confidence
"""

from pathlib import Path
import torch
from rfdetr import RFDETRNano

# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUM_CLASSES = 46
CLASS_NAMES = [f"class_{i}" for i in range(NUM_CLASSES)]

DATASET_DIR = Path("../../data/fashionpedia_coco")
OUTPUT_DIR = Path("../../models")

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
    Create RF-DETR Nano model.
    """
    if checkpoint_path:
        # FIX: Use 'pretrain_weights' instead of 'checkpoint'
        # rfdetr automatically handles loading the file and extracting EMA weights!
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
):
    """
    Train RF-DETR Nano on Fashionpedia dataset.
    """

    model = create_model(device)

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
    # Simply initialize using the factory.
    # Delete the manual torch.load() and load_state_dict() code!
    model = create_model(device, checkpoint_path)

    return model


# --------------------------------------------------
# Example usage
# --------------------------------------------------

if __name__ == "__main__":
    train_model()