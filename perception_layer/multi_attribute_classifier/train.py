"""
1.B — Multi-Attribute Classifier: Training
===========================================
Trains the MultiHeadAttributeClassifier on pseudo-labeled Fashionpedia crops.

Loss design
-----------
  fit / style       : KL-divergence with soft pseudo-labels
  weather_warmth    : SmoothL1Loss scalar regression
  formality_score   : SmoothL1Loss scalar regression

  total = λ_fit * L_fit + λ_style * L_style
        + λ_weather * L_weather + λ_formality * L_formality

Important:
    Your Fashionpedia COCO generator remaps categories to 0..26.
    Therefore, num_categories should be 27.

Usage
-----
    python train.py \
        --train_dir attribute_dataset/train \
        --valid_dir attribute_dataset/valid \
        --output_dir attribute_predictor/1b \
        --epochs 30 \
        --batch_size 64 \
        --lr 3e-4 \
        --num_categories 27
"""

import json
import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from perception_layer.multi_attribute_classifier.model import MultiHeadAttributeClassifier
from perception_layer.multi_attribute_classifier.attribute_dataset import AttributeDataset


# ─────────────────────────────────────────────────────
# Reproducibility / device
# ─────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────

class MultiHeadLoss(nn.Module):
    def __init__(
        self,
        lambda_fit: float = 1.0,
        lambda_style: float = 1.0,
        lambda_weather: float = 1.5,
        lambda_formality: float = 1.5,
        label_temperature: float = 0.7,
    ):
        super().__init__()

        self.lw = {
            "fit": lambda_fit,
            "style": lambda_style,
            "weather": lambda_weather,
            "formality": lambda_formality,
        }

        self.label_temperature = label_temperature

        # More robust than MSE for noisy pseudo-labels.
        self.reg_loss = nn.SmoothL1Loss(beta=0.1)

    def sharpen(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Slightly sharpens FashionCLIP soft labels.
        This helps the student learn clearer targets without fully hard-labeling them.
        """
        probs = probs.clamp_min(1e-8)
        probs = probs ** (1.0 / self.label_temperature)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return probs

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        fit_target = self.sharpen(batch["fit"])
        style_target = self.sharpen(batch["style"])

        l_fit = F.kl_div(
            F.log_softmax(pred["fit"], dim=-1),
            fit_target,
            reduction="batchmean",
        )

        l_style = F.kl_div(
            F.log_softmax(pred["style"], dim=-1),
            style_target,
            reduction="batchmean",
        )

        l_weather = self.reg_loss(
            pred["weather_warmth"],
            batch["weather_warmth"],
        )

        l_formality = self.reg_loss(
            pred["formality_score"],
            batch["formality_score"],
        )

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

def argmax_accuracy(logits: torch.Tensor, soft_targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    true = soft_targets.argmax(dim=-1)
    return (pred == true).float().mean().item()


def scalar_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred - target).abs().mean().item()


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    non_blocking = device == "cuda"

    return {
        key: value.to(device, non_blocking=non_blocking)
        for key, value in batch.items()
    }


# ─────────────────────────────────────────────────────
# Epoch runner
# ─────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiHeadLoss,
    optimizer: Optional[AdamW],
    device: str,
    train: bool = True,
    amp: bool = False,
) -> dict[str, float]:
    model.train(mode=train)

    totals = {
        "loss": 0.0,
        "loss_fit": 0.0,
        "loss_style": 0.0,
        "loss_weather": 0.0,
        "loss_formality": 0.0,
        "acc_fit": 0.0,
        "acc_style": 0.0,
        "mae_weather": 0.0,
        "mae_formality": 0.0,
    }

    total_samples = 0

    use_amp = amp and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Modern autocast requires a device_type string ('cuda' or 'cpu')
    device_type = "cuda" if device == "cuda" else "cpu"
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch in tqdm(loader, leave=False):
            batch = move_batch_to_device(batch, device)

            imgs = batch["image"]
            cats = batch["category_id"]
            batch_size = imgs.size(0)

            with torch.autocast(device_type=device_type, enabled=use_amp):
                pred = model(imgs, cats)
                loss, bkd = criterion(pred, batch)

            if train:
                if optimizer is None:
                    raise RuntimeError("Optimizer is required when train=True")

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            # Accumulate metrics
            totals["loss"] += loss.item() * batch_size
            totals["loss_fit"] += bkd["fit"] * batch_size
            totals["loss_style"] += bkd["style"] * batch_size
            totals["loss_weather"] += bkd["weather"] * batch_size
            totals["loss_formality"] += bkd["formality"] * batch_size

            totals["acc_fit"] += argmax_accuracy(pred["fit"], batch["fit"]) * batch_size
            totals["acc_style"] += argmax_accuracy(pred["style"], batch["style"]) * batch_size
            totals["mae_weather"] += scalar_mae(pred["weather_warmth"], batch["weather_warmth"]) * batch_size
            totals["mae_formality"] += scalar_mae(pred["formality_score"], batch["formality_score"]) * batch_size

            total_samples += batch_size

    total_samples = max(total_samples, 1)

    return {
        key: value / total_samples
        for key, value in totals.items()
    }


# ─────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    valid_metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "valid_metrics": valid_metrics,
        "config": vars(args),
    }

    torch.save(checkpoint, path)


# ─────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    device = get_best_device()
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(log_dir))

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train_ds = AttributeDataset(args.train_dir)
    valid_ds = AttributeDataset(args.valid_dir)

    pin_memory = device == "cuda"
    persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    model = MultiHeadAttributeClassifier(
        num_categories=args.num_categories,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * args.min_lr_ratio,
    )

    criterion = MultiHeadLoss(
        lambda_fit=args.lambda_fit,
        lambda_style=args.lambda_style,
        lambda_weather=args.lambda_weather,
        lambda_formality=args.lambda_formality,
        label_temperature=args.label_temperature,
    )

    best_loss = float("inf")
    patience_counter = 0
    history = []

    try:
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")

            train_m = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train=True,
                amp=args.amp,
            )

            valid_m = run_epoch(
                model=model,
                loader=valid_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                train=False,
                amp=False,
            )

            scheduler.step()

            for k, v in train_m.items():
                writer.add_scalar(f"train/{k}", v, epoch)

            for k, v in valid_m.items():
                writer.add_scalar(f"valid/{k}", v, epoch)

            writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

            print(
                f"Train loss {train_m['loss']:.4f} | "
                f"Valid loss {valid_m['loss']:.4f} | "
                f"Fit acc {valid_m['acc_fit']:.4f} | "
                f"Style acc {valid_m['acc_style']:.4f} | "
                f"Weather MAE {valid_m['mae_weather']:.4f} | "
                f"Formality MAE {valid_m['mae_formality']:.4f}"
            )

            epoch_record = {
                "epoch": epoch,
                "train": train_m,
                "valid": valid_m,
                "lr": scheduler.get_last_lr()[0],
            }

            history.append(epoch_record)

            # Always save latest checkpoint.
            save_checkpoint(
                path=output_dir / "last_checkpoint.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                valid_metrics=valid_m,
                args=args,
            )

            # Optional epoch checkpoint.
            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoint(
                    path=output_dir / f"epoch_{epoch:03d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    valid_metrics=valid_m,
                    args=args,
                )

            # Best model.
            if valid_m["loss"] < best_loss - args.min_delta:
                best_loss = valid_m["loss"]
                patience_counter = 0

                save_checkpoint(
                    path=output_dir / "best_checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    valid_metrics=valid_m,
                    args=args,
                )

                # Inference-compatible state_dict only.
                torch.save(
                    model.state_dict(),
                    output_dir / "best_model.pt",
                )

                print("✓ New best model saved")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{args.patience})")

            with open(output_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            if patience_counter >= args.patience:
                print("Early stopping triggered")
                break

    finally:
        # Ensures resources are freed safely, even if interrupted.
        writer.close()

    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--valid_dir", required=True)

    parser.add_argument("--output_dir", default="attribute_predictor/1b")
    parser.add_argument("--log_dir", default=None)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)

    parser.add_argument("--num_workers", type=int, default=4)

    # IMPORTANT:
    # Your Fashionpedia COCO generator remaps categories to 0..26.
    parser.add_argument("--num_categories", type=int, default=27)

    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--lambda_fit", type=float, default=1.0)
    parser.add_argument("--lambda_style", type=float, default=1.0)
    parser.add_argument("--lambda_weather", type=float, default=1.5)
    parser.add_argument("--lambda_formality", type=float, default=1.5)

    parser.add_argument("--label_temperature", type=float, default=0.7)

    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)

    # Use only on CUDA. It is ignored on CPU/MPS.
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()
    train(args)