import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from perception_layer import color_utils
from model import OutfitEmbeddingTransformer


class TripletFashionDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.outfits = data["outfits"]
        self.pool = data["pool"]

    def __len__(self):
        return len(self.outfits)

    def __getitem__(self, idx):
        anchor = self.outfits[idx]
        a_v = torch.stack(anchor["vecs"])
        a_c = torch.tensor(anchor["cats"])

        # Negative Generation: Swap one item to create a clash
        neg_vecs = [v for v in anchor["vecs"]]
        neg_cats = [c for c in anchor["cats"]]
        swap_idx = random.randrange(len(neg_vecs))

        # Hard Negative Mining: Ensure color score < 0.3
        for _ in range(5):
            candidate = random.choice(self.pool)
            temp_imgs = list(anchor["crops"])
            temp_imgs[swap_idx] = candidate["img"]
            if color_utils.harmony_score_from_images(temp_imgs) < 0.3:
                neg_vecs[swap_idx], neg_cats[swap_idx] = candidate["vec"], candidate["cat"]
                break

        return {"a_v": a_v, "a_c": a_c, "n_v": torch.stack(neg_vecs), "n_c": torch.tensor(neg_cats)}


def train(data_path, epochs=100, patience=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OutfitEmbeddingTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    writer = SummaryWriter("runs/stylist_v1")

    loader = DataLoader(TripletFashionDataset(data_path), batch_size=32, shuffle=True)
    best_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for b in loader:
            optimizer.zero_grad()
            # In Triplet, Anchor and Positive are the same (Ground Truth)
            emb_a = model(b['a_v'].to(device), b['a_c'].to(device))
            emb_n = model(b['n_v'].to(device), b['n_c'].to(device))

            # Loss pushes ground truth (emb_a) away from clashing mix (emb_n)
            loss = criterion(emb_a, emb_a, emb_n)
            loss.backward();
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch {epoch}: Loss {avg_loss:.4f}")

        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss;
            counter = 0
            torch.save(model.state_dict(), "best_stylist.pth")
        else:
            counter += 1
            if counter >= patience: break