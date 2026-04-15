"""
train.py

Command-line training entrypoint for RF-DETR Nano
trained on Fashionpedia.

Usage:
    python train.py
    python train.py --epochs 50
    python train.py --batch_size 8
    python train.py --resume models/last.pth
"""

import argparse
from pathlib import Path
import torch

from perception_layer.clothing_detection_segmintation.model import create_model, TRAIN_CONFIG, DEVICE


# --------------------------------------------------
# Argument Parser
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train RF-DETR Nano")

    parser.add_argument("--dataset_dir", type=str, default="../../data/fashionpedia_coco")
    parser.add_argument("--output_dir", type=str, default="../../models")

    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["lr"])
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


# --------------------------------------------------
# Training Runner
# --------------------------------------------------

def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = DEVICE
    model = create_model(device)

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)

    print("\nTraining Configuration")
    print("----------------------")
    print(f"Device:        {device}")
    print(f"Dataset:       {dataset_dir}")
    print(f"Output:        {output_dir}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("----------------------\n")

    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        **{k: v for k, v in TRAIN_CONFIG.items() if k not in ["batch_size", "epochs", "lr"]},
    )


if __name__ == "__main__":
    main()