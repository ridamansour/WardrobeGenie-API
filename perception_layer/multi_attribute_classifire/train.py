"""
1.B — Multi-Attribute Classifier: Training
===========================================
Trains the MultiHeadAttributeClassifier on pseudo-labeled Fashionpedia crops.

Loss design
-----------
  fit / style       : KL-divergence  (soft distribution matching)
  weather_warmth    : MSE            (scalar regression)
  formality_score   : MSE            (scalar regression)

  total = λ_fit * L_fit + λ_style * L_style
        + λ_weather * L_weather + λ_formality * L_formality

Usage
-----
    python train.py \
        --train_dir  attribute_dataset/train \
        --valid_dir  attribute_dataset/valid \
        --output_dir runs/1b \
        --epochs 30 \
        --batch_size 64 \
        --lr 3e-4
"""

import os
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from perception_layer.multi_attribute_classifire.model import MultiHeadAttributeClassifier
from perception_layer.multi_attribute_classifire.attribute_dataset import AttributeDataset


# ─────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────

class MultiHeadLoss(nn.Module):
    def __init__(
        self,
        lambda_fit=1.0,
        lambda_style=1.0,
        lambda_weather=2.0,
        lambda_formality=2.0,
    ):
        super().__init__()
        self.lw = {
            "fit": lambda_fit,
            "style": lambda_style,
            "weather": lambda_weather,
            "formality": lambda_formality,
        }
        self.mse = nn.MSELoss()

    def forward(self, pred, batch):
        l_fit = F.kl_div(
            F.log_softmax(pred["fit"], dim=-1),
            batch["fit"],
            reduction="batchmean",
        )

        l_style = F.kl_div(
            F.log_softmax(pred["style"], dim=-1),
            batch["style"],
            reduction="batchmean",
        )

        l_weather = self.mse(pred["weather_warmth"], batch["weather_warmth"])
        l_formality = self.mse(pred["formality_score"], batch["formality_score"])

        total = (
            self.lw["fit"] * l_fit
            + self.lw["style"] * l_style
            + self.lw["weather"] * l_weather
            + self.lw["formality"] * l_formality
        )

        return total, {
            "fit": l_fit.item(),
            "style": l_style.item(),
            "weather": l_weather.item(),
            "formality": l_formality.item(),
        }


# ─────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────

def argmax_accuracy(logits, soft_targets):
    pred = logits.argmax(dim=-1)
    true = soft_targets.argmax(dim=-1)
    return (pred == true).float().mean().item()


def scalar_mae(pred, target):
    return (pred - target).abs().mean().item()


# ─────────────────────────────────────────────────────
# Epoch runner
# ─────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0
    breakdown = {k: 0 for k in ["fit", "style", "weather", "formality"]}
    acc_fit = acc_style = mae_w = mae_f = 0
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in tqdm(loader, leave=False):
            imgs = batch["image"].to(device)
            cats = batch["category_id"].to(device)

            targets = {
                "fit": batch["fit"].to(device),
                "style": batch["style"].to(device),
                "weather_warmth": batch["weather_warmth"].to(device),
                "formality_score": batch["formality_score"].to(device),
            }

            pred = model(imgs, cats)
            loss, bkd = criterion(pred, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            for k in breakdown:
                breakdown[k] += bkd[k]

            acc_fit += argmax_accuracy(pred["fit"], targets["fit"])
            acc_style += argmax_accuracy(pred["style"], targets["style"])
            mae_w += scalar_mae(pred["weather_warmth"], targets["weather_warmth"])
            mae_f += scalar_mae(pred["formality_score"], targets["formality_score"])

            n += 1

    N = max(n, 1)
    return {
        "loss": total_loss / N,
        "loss_fit": breakdown["fit"] / N,
        "loss_style": breakdown["style"] / N,
        "loss_weather": breakdown["weather"] / N,
        "loss_formality": breakdown["formality"] / N,
        "acc_fit": acc_fit / N,
        "acc_style": acc_style / N,
        "mae_weather": mae_w / N,
        "mae_formality": mae_f / N,
    }


# ─────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    writer = SummaryWriter(args.log_dir)

    train_ds = AttributeDataset(args.train_dir)
    valid_ds = AttributeDataset(args.valid_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = MultiHeadAttributeClassifier(
        num_categories=args.num_categories
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = MultiHeadLoss()

    best_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_m = run_epoch(model, train_loader, criterion, optimizer, device, True)
        valid_m = run_epoch(model, valid_loader, criterion, None, device, False)

        scheduler.step()

        # ─── TensorBoard ───
        for k, v in train_m.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in valid_m.items():
            writer.add_scalar(f"valid/{k}", v, epoch)

        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        print(f"Train loss {train_m['loss']:.4f} | Valid loss {valid_m['loss']:.4f}")

        # ─── Save checkpoint each epoch ───
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "valid_loss": valid_m["loss"],
        }, Path(args.output_dir) / f"epoch_{epoch}.pt")

        # ─── Best model ───
        if valid_m["loss"] < best_loss - args.min_delta:
            best_loss = valid_m["loss"]
            patience_counter = 0

            torch.save(model.state_dict(),
                       Path(args.output_dir) / "best_model.pt")

            print("✓ New best model saved")

        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{args.patience})")

        if patience_counter >= args.patience:
            print("Early stopping triggered")
            break

        history.append({"epoch": epoch, "train": train_m, "valid": valid_m})

    writer.close()

    with open(Path(args.output_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--valid_dir", required=True)
    parser.add_argument("--output_dir", default="runs/1b")
    parser.add_argument("--log_dir", default="runs/tensorboard")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_categories", type=int, default=46)

    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min_delta", type=float, default=1e-4)

    train(parser.parse_args())