"""
finetune_clip.py
Fine-tunes CLIP on the manually labeled dataset using Contrastive Learning.
"""

import os
import json
import torch
import argparse
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torch.optim import AdamW
from tqdm import tqdm

PROMPT_MAP = {
    "fit": {
        "Slim": "a photo of a garment with a tight, slim fit",
        "Regular": "a photo of a garment with a regular, standard fit",
        "Oversized": "a photo of a garment with an oversized, baggy fit"
    },
    "style": {
        "Formal": "a photo of a formal style garment",
        "Casual": "a photo of a casual style garment",
        "Sport": "a photo of a sport style garment",
        "Streetwear": "a photo of a streetwear style garment"
    },
    "weather": {
        "Winter (Freezing)": "a photo of someone wearing a garment perfect for freezing winter weather",
        "Transitional (Mild)": "a photo of someone wearing a garment perfect for mild, layered transitional weather",
        "Summer (Hot)": "a photo of someone wearing a garment perfect for hot, sunny summer weather"
    },
    "formality": {
        "Black Tie": "a photo of someone wearing a garment perfect for a formal black tie event",
        "Business": "a photo of someone wearing a garment perfect for a professional business office",
        "Smart Casual": "a photo of someone wearing a garment perfect for a smart casual dinner",
        "Everyday Casual": "a photo of someone wearing a garment perfect for everyday casual wear",
        "Gym/Workout": "a photo of someone wearing a garment perfect for working out at the gym"
    }
}


class ManualClipDataset(Dataset):
    def __init__(self, data_dir: str, processor: CLIPProcessor):
        self.data_dir = Path(data_dir)
        self.processor = processor
        with open(self.data_dir / "manual_labels.json", "r") as f:
            self.labels = json.load(f)
        self.image_files = list(self.labels.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(self.data_dir / img_name).convert("RGB")
        lbl = self.labels[img_name]

        text_desc = (
            f"{PROMPT_MAP['fit'][lbl['fit']]}. "
            f"{PROMPT_MAP['style'][lbl['style']]}. "
            f"{PROMPT_MAP['weather'][lbl['weather']]}. "
            f"{PROMPT_MAP['formality'][lbl['formality']]}."
        )

        inputs = self.processor(
            text=text_desc, images=image, return_tensors="pt",
            padding="max_length", truncation=True, max_length=77
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}


def finetune_clip(args):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name, use_fast=False)

    dataset = ManualClipDataset(args.data_dir, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(**batch)
            logits_img = outputs.logits_per_image
            logits_txt = outputs.logits_per_text

            targets = torch.arange(logits_img.shape[0]).to(device)
            loss = (loss_fn(logits_img, targets) + loss_fn(logits_txt, targets)) / 2

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Fine-tuned model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="manual_labeling_data")
    parser.add_argument("--output_dir", default="fine_tuned_clip")
    parser.add_argument("--model_name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    finetune_clip(parser.parse_args())