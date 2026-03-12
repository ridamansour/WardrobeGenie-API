import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import OutfitSetTransformer


def train_model(train_loader, val_loader, model, epochs=20, log_dir="logs/stylist_run"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    writer = SummaryWriter(log_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    checkpoint_path = "checkpoints/best_stylist_model.pth"
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for item_vecs, cat_ids, targets in train_loader:
            item_vecs, cat_ids, targets = item_vecs.to(device), cat_ids.to(device), targets.to(device)

            optimizer.zero_grad()
            preds = model(item_vecs, cat_ids)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for item_vecs, cat_ids, targets in val_loader:
                item_vecs, cat_ids, targets = item_vecs.to(device), cat_ids.to(device), targets.to(device)
                preds = model(item_vecs, cat_ids)
                val_loss += criterion(preds, targets).item()

        avg_val_loss = val_loss / len(val_loader)

        # Logging to TensorBoard
        writer.add_scalars('Loss', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)
        print(f"Epoch {epoch} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, checkpoint_path)
            print(f"--> Saved New Best Model at Epoch {epoch}")

    writer.close()