import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # New: TensorBoard
from model import StudentEncoder  #
from dataset import FashionpediaCropDataset, collate_fn  #


def train_model(train_dir, valid_dir, epochs=50, batch_size=128, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir="runs/student_distillation")

    # Load pre-computed dataset
    train_ds = FashionpediaCropDataset(train_dir)
    valid_ds = FashionpediaCropDataset(valid_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size,
                              collate_fn=collate_fn, num_workers=4)

    # Model and Optimization
    student = StudentEncoder().to(device)
    optimizer = optim.AdamW(student.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    # 2. Early Stopping Variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    print(f"Starting Training with Early Stopping (Patience: {patience})...")

    for epoch in range(epochs):
        # --- Training Phase ---
        student.train()
        train_loss = 0.0
        for batch in train_loader:
            if batch is None: continue
            imgs, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            preds = student(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                if batch is None: continue
                imgs, targets = batch[0].to(device), batch[1].to(device)
                preds = student(imgs)
                loss = criterion(preds, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)

        # 3. Log to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        print(f"Epoch {epoch + 1:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # 4. Early Stopping & Checkpointing Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(student.state_dict(), "best_student_model.pth")
            print("  --> New best model saved!")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    writer.close()
    print("Training Complete.")


if __name__ == "__main__":
    train_model(
        train_dir="../../data/fashionpedia_coco/train",
        valid_dir="../../data/fashionpedia_coco/valid"
    )