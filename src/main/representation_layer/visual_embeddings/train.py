import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import StudentEncoder
from dataset import FashionpediaCropDataset, collate_fn

from src.config import FASHIONOPEDIA_COCO_DIR, VISUAL_EMBEDDER_DIR


def train_model(train_dir, valid_dir, output_dir, epochs=50, batch_size=128, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "runs"), exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs/student_distillation"))

    # Load datasets
    train_ds = FashionpediaCropDataset(train_dir)
    valid_ds = FashionpediaCropDataset(valid_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)

    student = StudentEncoder().to(device)
    optimizer = optim.AdamW(student.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    print(f"Starting Training on {device} (Patience: {patience})...")

    for epoch in range(epochs):
        # --- Training Phase ---
        student.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for batch in train_pbar:
            if batch is None: continue
            imgs, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
            preds = student(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        student.eval()
        val_loss = 0.0
        val_pbar = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{epochs} [Valid]")

        with torch.no_grad():
            for batch in val_pbar:
                if batch is None: continue
                imgs, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                preds = student(imgs)
                loss = criterion(preds, targets)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(valid_loader)

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        print(f"--> Epoch {epoch + 1:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # --- Early Stopping & Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            model_path = os.path.join(output_dir, "best_student_model.pth")
            torch.save(student.state_dict(), model_path)
            print(f"    New best model saved to {model_path}!")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    writer.close()
    print("Training Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Student Visual Embedder")
    parser.add_argument("--train_dir", default=FASHIONOPEDIA_COCO_DIR / "train", type=str)
    parser.add_argument("--valid_dir", default=FASHIONOPEDIA_COCO_DIR / "valid", type=str)
    parser.add_argument("--output_dir", default=VISUAL_EMBEDDER_DIR, type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    args = parser.parse_args()

    train_model(
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
    )