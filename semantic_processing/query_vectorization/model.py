import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# CHANGED: Using CLIPTextModelWithProjection to get the correctly projected fashion text embeddings
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, DistilBertModel, DistilBertTokenizer
import torch.nn.functional as F
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"


# ---------------------------------------------------------
# 1. Define the Student Model (Distilled BERT -> 512 dim)
# ---------------------------------------------------------
class DistilledQueryEncoder(nn.Module):
    """MobileNetV3-style sister text network using DistilBERT."""

    def __init__(self, base_model="distilbert-base-uncased", out_dim=512):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(base_model)
        # Projection head to map BERT's 768-dim output to FashionCLIP's 512-dim space
        self.projection = nn.Linear(self.bert.config.hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token equivalent (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_output)
        # Normalize to match CLIP's normalized embedding space
        return F.normalize(projected, dim=-1)


# ---------------------------------------------------------
# 2. Dataset Setup
# ---------------------------------------------------------
class QueryDataset(Dataset):
    def __init__(self, queries):
        self.queries = queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx]


# ---------------------------------------------------------
# 3. Distillation Loop
# ---------------------------------------------------------
def train_distillation(queries, epochs=5, batch_size=32, lr=2e-5):
    # CHANGED: Initialize Teacher with FashionCLIP instead of OpenAI CLIP
    teacher_name = "patrickjohncyh/fashion-clip"
    print(f"Loading FashionCLIP Teacher text components from {teacher_name}...")
    teacher_tokenizer = CLIPTokenizer.from_pretrained(teacher_name)
    teacher_model = CLIPTextModelWithProjection.from_pretrained(teacher_name).to(device)
    teacher_model.eval()

    # Freeze teacher parameters
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Initialize Student
    student_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    student_model = DistilledQueryEncoder().to(device)

    optimizer = optim.AdamW(student_model.parameters(), lr=lr)
    dataloader = DataLoader(QueryDataset(queries), batch_size=batch_size, shuffle=True)

    print("Starting Distillation...")
    for epoch in range(epochs):
        student_model.train()
        total_loss = 0

        for batch_queries in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            # 1. Get Teacher Embeddings (No gradients needed)
            with torch.no_grad():
                t_tokens = teacher_tokenizer(batch_queries, padding=True, truncation=True, return_tensors="pt").to(
                    device)
                # CHANGED: extract 'text_embeds' which includes the 512-dim fashion projection
                t_features = teacher_model(**t_tokens).text_embeds
                t_features = F.normalize(t_features, dim=-1)

            # 2. Get Student Embeddings
            s_tokens = student_tokenizer(batch_queries, padding=True, truncation=True, return_tensors="pt").to(device)
            s_features = student_model(s_tokens['input_ids'], s_tokens['attention_mask'])

            # 3. Calculate Loss (Cosine Embedding Loss)
            target = torch.ones(s_features.size(0)).to(device)
            loss = F.cosine_embedding_loss(s_features, t_features, target)

            # 4. Backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} | Average Loss: {total_loss / len(dataloader):.4f}")

    # Ensure output directory exists before saving
    output_path = "../../models/nlp_query/distilled_query_encoder.pth"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.save(student_model.state_dict(), output_path)
    print(f"Saved distilled model to {output_path}")

    return student_model, student_tokenizer


if __name__ == "__main__":
    # Load the massive synthetic dataset
    print("Loading synthetic queries...")
    with open("../../data/fashion_queries_realworld.txt", "r") as f:
        synthetic_queries = [line.strip() for line in f.readlines()]

    # Train the student model
    train_distillation(synthetic_queries, epochs=5, batch_size=64, lr=2e-5)