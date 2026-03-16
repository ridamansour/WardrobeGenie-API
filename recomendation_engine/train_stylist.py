import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from perception_layer import color_utils
from model import OutfitEmbeddingTransformer


class TripletFashionDataset(Dataset):
    def __init__(self, data_path):
        # ---------------------------------------------------------
        # FIX: Force all saved GPU tensors into CPU memory on load
        # ---------------------------------------------------------
        data = torch.load(data_path, map_location='cpu')
        self.outfits = data["outfits"]
        self.pool = data["pool"]

    def __len__(self):
        return len(self.outfits)

    def __getitem__(self, idx):
        anchor = self.outfits[idx]

        # Flatten vectors to prevent PyTorch broadcasting shape mismatches
        a_v = torch.stack(anchor["vecs"]).view(-1, 512)
        a_c = torch.tensor(anchor["cats"])

        neg_vecs = [v for v in anchor["vecs"]]
        neg_cats = [c for c in anchor["cats"]]
        swap_idx = random.randrange(len(neg_vecs))

        for _ in range(5):
            candidate = random.choice(self.pool)

            # Fast, math-only validation using pre-computed colors (No Image.open!)
            temp_colors = list(anchor["colors"])
            temp_colors[swap_idx] = candidate["color_data"]

            if color_utils.harmony_score_from_precomputed(temp_colors) < 0.3:
                neg_vecs[swap_idx] = candidate["vec"]
                neg_cats[swap_idx] = candidate["cat"]
                break

        return {
            "a_v": a_v, "a_c": a_c,
            "n_v": torch.stack(neg_vecs).view(-1, 512), "n_c": torch.tensor(neg_cats)
        }


def collate_fn(batch):
    a_v = pad_sequence([item['a_v'] for item in batch], batch_first=True, padding_value=0.0)
    a_c = pad_sequence([item['a_c'] for item in batch], batch_first=True, padding_value=0)
    n_v = pad_sequence([item['n_v'] for item in batch], batch_first=True, padding_value=0.0)
    n_c = pad_sequence([item['n_c'] for item in batch], batch_first=True, padding_value=0)

    # Force identical shapes to prevent addition mismatch in transformer
    max_len_a = max(a_v.shape[1], a_c.shape[1])
    max_len_n = max(n_v.shape[1], n_c.shape[1])

    if a_v.shape[1] < max_len_a:
        a_v = torch.cat([a_v, torch.zeros(a_v.shape[0], max_len_a - a_v.shape[1], a_v.shape[2])], dim=1)
    if a_c.shape[1] < max_len_a:
        a_c = torch.cat([a_c, torch.zeros(a_c.shape[0], max_len_a - a_c.shape[1], dtype=a_c.dtype)], dim=1)

    if n_v.shape[1] < max_len_n:
        n_v = torch.cat([n_v, torch.zeros(n_v.shape[0], max_len_n - n_v.shape[1], n_v.shape[2])], dim=1)
    if n_c.shape[1] < max_len_n:
        n_c = torch.cat([n_c, torch.zeros(n_c.shape[0], max_len_n - n_c.shape[1], dtype=n_c.dtype)], dim=1)

    return {'a_v': a_v, 'a_c': a_c, 'n_v': n_v, 'n_c': n_c}


def train(data_path, epochs=100, patience=7, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = OutfitEmbeddingTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    writer = SummaryWriter("runs/stylist_v1")

    full_dataset = TripletFashionDataset(data_path)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # multi-worker loading for GPU speed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4,
                            pin_memory=True)

    best_val_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for b in train_loader:
            optimizer.zero_grad()
            emb_a = model(b['a_v'].to(device), b['a_c'].to(device))
            emb_n = model(b['n_v'].to(device), b['n_c'].to(device))
            loss = criterion(emb_a, emb_a, emb_n)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for b in val_loader:
                emb_a = model(b['a_v'].to(device), b['a_c'].to(device))
                emb_n = model(b['n_v'].to(device), b['n_c'].to(device))
                loss = criterion(emb_a, emb_a, emb_n)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), "best_stylist.pth")
            print("  -> Saved new best model")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}! Best Val Loss: {best_val_loss:.4f}")
                break


if __name__ == "__main__":
    train("../data/fashionpedia_coco/processed_pool.pt")