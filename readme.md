# The Project Design
## 1. Perception Layer (Image Processing)
These components process raw user photos to detect clothing items and extract attributes. Where supported by the platform, models can run in parallel using threads.
### A. Clothing Detection & Segmentation (RF-DETR Segmentation Nano)
**Model:** RF-DETR Segmentation Nano (Instance Segmentation), trained on Fashionpedia (≈50K images with masks).
**Purpose:** Detect garments (top/bottom/shoes), crop them, and provide category + confidence.
#### Input:
A raw photo of a person or a flat-lay clothing image.
#### Output:
A list of detected clothing items (bbox + mask + category + confidence + cropped image).
#### Example:
```txt
Input:
    PIL.Image()  # e.g. (1920x1080)
Output:
    [
        {
            "category_id": 1,         # "Top"
            "confidence": 0.98,
            "bbox": [100, 50, 400, 500],
            "mask": np.ndarray(...),  # optional binary mask
            "image": PIL.Image()      # Cropped Top
        },
        {
            "category_id": 2,         # "Bottom"
            "confidence": 0.94,
            "bbox": [120, 420, 420, 980],
            "mask": np.ndarray(...),
            "image": PIL.Image()
        }
    ]
```
---
### B. Multi-Attribute Classifier (EfficientNet-B0, Multi-Head)
**Architecture:** Single backbone + multiple heads.
**Why:** Run one forward pass and predict multiple attributes (saves compute vs separate models).
**Predicted attributes (examples):**
* fit: slim / regular / oversized
* style: formal / casual / sporty
* weather_warmth: 0–1 (cold → warm)
* formality_score: 0–1
#### Input:
A normalized cropped item image, optionally with the garment type id.
#### Output:
Attribute probabilities + scalar metrics.
#### Example:
```txt
Input:
    torch.Tensor()  # Shape: (1, 3, 224, 224)
Output:
    {
        "fit": torch.Tensor([0.1, 0.8, 0.1]),            # -> "Regular"
        "style": torch.Tensor([0.05, 0.9, 0.05]),        # -> "Casual"
        "weather_warmth": 0.75,                          # 0.0 cold -> 1.0 warm
        "formality_score": 0.20                          # 0.0 PJs  -> 1.0 tux
    }
```
**Labeling strategy:**
* Cloud training can use **pseudo-labeling** via a vision-language model (CLIP-style) + a small manually verified set.
---
### C. Color Quantizer (K-Means Clustering)
**Method:** K-Means on pixel colors (no deep learning).  
**Purpose:** Extract dominant colors for color theory / fuzzy logic.
#### Input:
Cropped clothing image from Step A.
#### Output:
List of dominant HEX codes and their percentage.
#### Example:
```txt
Input:
    PIL.Image()  # cropped shirt
Output:
    [
        ("#1A2B4C", 0.65),  # dominant
        ("#FFFFFF", 0.30),
        ("#FF0000", 0.05)
    ]
```
---
### D. Clothing Color Classifier (Function)
**Purpose:** Convert the quantizer output to a compact “dominant colors” list used by scoring.
This is only for training the set transformers to classify bad fits.
#### Input:
Output from the K-Means step.
#### Output:
Top dominant colors (e.g., those above a threshold).
#### Example:
```txt
Input:
    [("#FFFFFF", 0.40), ("#FF983F", 0.30), ("#000000", 0.10)]
Output:
    [
        ("#FFFFFF", 0.40),
        ("#FF983F", 0.30)
    ]
```
---
### E. On-Device Implementation (Mobile Deployment)
**Goal:** Run inference locally for privacy + offline operation.
* Convert models to **TFLite** (Android) or **CoreML** (iOS/macOS).
* Apply **quantization** to reduce size and accelerate inference.
**Note:** Inference time may be several seconds depending on device/model size.
---
## 2. Textual Query Understanding (Semantic Processing)
This layer transforms a user request (text) into intent signals and an embedding usable by ranking.
### A. Query Vectorization (Mobile Text Encoder)
**Model:** Distilled BERT / MobileCLIP text branch.
**Purpose:** Encode user query into a 512-dim vector aligned with image vectors.
#### Input:
A user query string.
#### Output:
A 512-dim embedding vector.
#### Example:
```txt
Input:
    "business meeting, looking for comfort"
Output:
    torch.Tensor()  # Shape: (1, 512)
```
---
### B. Intent & Filter Extraction (Zero-Shot Text Classification)
**Method:** Compare query against label prompts like “formal”, “casual”, “rainy”, “warm”, “office”.
**Purpose:** Convert “soft” text into “hard” filters (e.g., formality threshold).
#### Example:
```txt
Input:
    "business meeting"
Output:
    {
        "intent_formal": 0.88,
        "intent_office": 0.91,
        "intent_sporty": 0.05
    }
Derived Filter:
    formality_score >= 0.60
```
---
### C. Hybrid Filtering (Semantic + Boolean)
**Idea:** A combined score:
* semantic similarity (cosine)
* context constraints (weather/formality rules)
#### Example:
```txt
score_item = α * context_score + β * cosine(item_vec, query_vec)
```
---
## 3. Representation Layer (Vectorization)
This layer produces vectors for fast similarity search and outfit relevance.
### A. Visual Embeddings (Student CLIP)
**Model:** MobileNetV3 image encoder distilled from CLIP (ViT teacher).
**Purpose:** Convert each item image into a normalized 512-dim “vibe” embedding.
#### Input:
Cropped image tensor.
#### Output:
Normalized embedding.
#### Example:
```txt
Input:
    torch.Tensor()  # (1, 3, 224, 224)
Output:
    torch.Tensor()  # (1, 512)
```
---
### B. Text Embeddings
**Purpose:** Ensure text and image vectors live in the same embedding space (512-dim).
---
### C. Local Vector Store
**Storage:** SQLite + local files, or a compact ANN index.
**Purpose:** Nearest-neighbor retrieval without internet.
#### Example:
```txt
Stored:
    item_id -> embedding(512) + metadata
Search:
    top_k by cosine similarity
```
---
## 4. The “Brain” Layer (Recommendation Engine)
This is where filtering + combination + scoring produces ranked outfits.
### A. Context-Aware Pre-Filter (Gatekeeper)
**Goal:** Reduce the wardrobe to a candidate set (~50 items) before trying combinations.
#### Input:
* Context (occasion, weather, temperature, etc.)
* Query embedding
* Wardrobe database
#### Output:
Candidate items list.
#### Example:
```txt
Input:
    Context: {"occasion":"office", "temp":15}
    Query: "business meeting"
Operation:
    1) filter formality_score >= 0.60
    2) filter weather_warmth <= threshold for cold
    3) rank by cosine similarity to query_vec, take top_k
Output:
    ["uuid_top_1", "uuid_bottom_5", "uuid_shoes_2", ...]
```
---
### B. Outfit Compatibility Model (Set-Transformer “Stylist”)
**Model:** Small Set-Transformer (attention-based)
**Purpose:** Given a set of item vectors (+ category ids), output outfit compatibility score.
#### Input:
* item_vectors: (1, N, 512)
* category_ids: (1, N)
#### Output:
compatibility_score (0–1)
#### Example:
```txt
Input:
    item_vectors: torch.Tensor()  # (1, 3, 512) top/bottom/shoes
    category_ids: torch.Tensor()  # (1, 3) [0,1,2]
Output:
    0.94
```
---
### C. Final Ranking (Compatibility + Relevance)
Combine:
* Aesthetic score (transformer)
* Relevance score (cosine between outfit vector and query vector)
#### Example:
```txt
final_score = α * aesthetic_score + β * relevance_score
```
---
### D. Personalization (Reinforcement-lite / Bandit)
**Goal:** Adapt scoring weights (α/β) or last-layer parameters from feedback.
#### Input:
* user feedback (like/dislike/skip)
* context + query embedding
#### Output:
Updated weights / preference parameters.
#### Example:
```txt
Input:
    reward = +1 (liked)
    state = (query_vec, context)
Output:
    α := α + Δα
    β := β + Δβ
```
---
## 5. Training Strategy & Data
### A. Cloud Training
All heavy training is done in the cloud (GPU):
* RF-DETR segmentation (Fashionpedia)
* EfficientNet multi-head attribute classifier (pseudo-labeling + small manual set)
* Student CLIP distillation (teacher CLIP → student MobileNet)
* Outfit transformer (outfit compatibility ranking)
### B. On-Device Deployment
* Export trained weights
* Convert to TFLite/CoreML
* Quantize models
* Keep user data local
---
## 6. Data Storage Schema (Updated)
```json
{
  "item_id": "uuid-1234",
  "owner_id": "user-5678",
  "category": "top",
  "sub_category": "t-shirt",
  "attributes": {
    "fit": "regular",
    "style": "casual"
  },
  "metrics": {
    "formality": 0.30,
    "weather_warmth": 0.80,
    "color_dominance": ["#1A2B4C", "#FFFFFF"]
  },
  "embedding": [0.042, -0.11, 0.55, "..."],
  "last_worn": "2026-02-14",
  "wear_count": 15,
  "dirty_status": false
}
```
---
## 7. Final Pipeline Flow
### A. Wardrobe Update (Image Input)
1. Run segmentation/detection
2. Crop each garment
3. Predict attributes
4. Extract colors
5. Compute/store embeddings + metadata locally
### B. User Query
1. Encode query into embedding
2. Extract intent filters
### C. Candidate Filtering
1. Apply context filters
2. Apply similarity ranking to select candidates
### D. Outfit Generation
Combine candidates into plausible outfits (top + bottom + shoes).
### E. Scoring & Ranking
1. Compatibility score (transformer)
2. Relevance score (cosine)
3. Final weighted score
### F. Learning Loop
Update user preference weights based on feedback.
---
# Models training/data
## 1. Perception Layer (Image Processing)
### A. Clothing Detection & Segmentation (RF-DETR Segmentation Nano)
**Data source:** Fasionopedia Dataset
**Fasionopedia Dataset Example**:
```
Labels:
    PIL.Image()  # e.g. (1920x1080)
Target:
    [
        {
            "category_id": 1,         # "Top",
            "bbox": [100, 50, 400, 500],
        },
        {
            "category_id": 2,         # "Bottom"
            "bbox": [120, 420, 420, 980]
        }
    ]
```
- [x] Designed
- [ ] Trained
- [ ] Evaluated
### B. Multi-Attribute Classifier (EfficientNet-B0, Multi-Head)
**Attributes source (image from Fasionopedia Dataset passed through model A1 (RF-DETR Segmentation Nano):**
```python
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
device = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------------------
# Load CLIP
# --------------------------------------------------
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
# --------------------------------------------------
# Prompt Definitions (visually grounded & separable)
# --------------------------------------------------
weather_prompts = [
    "a person wearing heavy insulated winter clothing, thick coat, scarf and gloves",
    "a person wearing layered clothing suitable for cool weather, light jacket or sweater",
    "a person wearing light breathable summer clothing, short sleeves and thin fabrics"
]
formality_prompts = [
    "black tie formal evening wear, tuxedo or elegant evening gown",
    "business professional suit or formal office attire",
    "smart business casual outfit, neat and polished",
    "casual everyday clothing, relaxed and comfortable",
    "athletic sportswear or gym training outfit"
]
fit_prompts = [
    "tight slim fit clothing closely fitted to the body",
    "regular fit clothing with standard comfortable cut",
    "oversized baggy loose clothing with wide silhouette"
]
style_prompts = [
    "formal elegant fashion style",
    "casual everyday fashion style",
    "sport athletic activewear style",
    "streetwear urban fashion style"
]
# --------------------------------------------------
# Encode text prompts ONCE (important for speed & stability)
# --------------------------------------------------
def encode_text(prompts):
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features / features.norm(dim=-1, keepdim=True)
weather_text = encode_text(weather_prompts)
formality_text = encode_text(formality_prompts)
fit_text = encode_text(fit_prompts)
style_text = encode_text(style_prompts)
# --------------------------------------------------
# Attribute extraction function
# --------------------------------------------------
def extract_attributes(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    # ---------- Weather / Warmth ----------
    weather_logits = image_features @ weather_text.T
    weather_probs = weather_logits.softmax(dim=-1).cpu()[0]
    cold, mild, hot = weather_probs.tolist()
    weather_warmth = (
        0.0 * cold +
        0.5 * mild +
        1.0 * hot
    )
    # ---------- Formality ----------
    formality_logits = image_features @ formality_text.T
    formality_probs = formality_logits.softmax(dim=-1).cpu()[0]
    formality_weights = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    formality_score = float((formality_probs * formality_weights).sum())
    # ---------- Fit ----------
    fit_logits = image_features @ fit_text.T
    fit_probs = fit_logits.softmax(dim=-1).cpu()[0]
    # ---------- Style ----------
    style_logits = image_features @ style_text.T
    style_probs = style_logits.softmax(dim=-1).cpu()[0]
    # --------------------------------------------------
    return {
        "fit": fit_probs.tolist(),               # [slim, regular, oversized]
        "style": style_probs.tolist(),           # [formal, casual, athletic, streetwear]
        "weather_warmth": float(weather_warmth), # 0=cold → 1=hot
        "formality_score": float(formality_score)
    }
# --------------------------------------------------
# Example usage
# --------------------------------------------------
image = Image.open("example.jpg").convert("RGB")
attributes = extract_attributes(image)
print(attributes)
```
**Target dataset**:
```
Labels:
    torch.Tensor()  # Shape: (1, 3, 224, 224)
Targets:
    {
        "fit": torch.Tensor([0.1, 0.8, 0.1]),            # -> "Regular"
        "style": torch.Tensor([0.05, 0.9, 0.05]),        # -> "Casual"
        "weather_warmth": 0.75,                          # 0.0 cold -> 1.0 warm
        "formality_score": 0.20                          # 0.0 PJs  -> 1.0 tux
    }
```
### C. Color Quantizer (K-Means Clustering) + D. Clothing Color Classifier (Function)
Implementation:
```python
import numpy as np
from sklearn.cluster import KMeans
def quantize_colors(pil_img, k=3, resize=120):
    """
    Extract dominant colors using K-Means clustering.
    Parameters
    ----------
    pil_img : PIL.Image
        Cropped clothing image.
    k : int
        Number of dominant colors to extract.
    resize : int
        Resize dimension for speed.
    Returns
    -------
    list of tuples
        [(hex_color, percentage), ...]
    """
    img = pil_img.copy()
    img.thumbnail((resize, resize))
    img_data = np.array(img)
    if len(img_data.shape) == 2:
        img_data = np.stack([img_data]*3, axis=-1)
    # Handle RGBA → RGB
    if img_data.shape[-1] == 4:
        img_data = img_data[:, :, :3]
    pixels = img_data.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(labels)
    percentages = counts / counts.sum()
    sorted_idx = np.argsort(percentages)[::-1]
    dominant_colors = []
    for i in sorted_idx:
        rgb = tuple(centers[i])
        hex_code = '#{:02X}{:02X}{:02X}'.format(*rgb)
        pct = float(percentages[i])
        dominant_colors.append((hex_code, round(pct, 4)))
    return dominant_colors
```
## 2. Textual Query Understanding (Semantic Processing)
### A. Query Vectorization
Implementation:
```python
import torch
from transformers import CLIPTokenizer, CLIPTextModel
device = "cuda" if torch.cuda.is_available() else "cpu"
class QueryVectorizer:
    """
    Encodes text queries into normalized 512-dim vectors
    aligned with CLIP image embeddings.
    """
    def __init__(self, model_name="apple/mobileclip-text-encoder"):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name).to(device)
        self.model.eval()
    def encode(self, query: str):
        tokens = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)
        with torch.no_grad():
            features = self.model(**tokens).pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
        return features
```
### B. Intent & Filter Extraction (Zero-Shot)
Implementation:
```python
import torch
import QueryVectorizer
class IntentExtractor:
    """
    Zero-shot intent detection using MobileCLIP text similarity.
    """
    def __init__(self, vectorizer: QueryVectorizer):
        self.vectorizer = vectorizer
        self.intent_labels = [
            "formal office business setting",
            "casual everyday wear",
            "sport athletic activity",
            "cold winter weather",
            "hot summer weather",
            "rainy wet conditions",
            "comfortable relaxed clothing"
        ]
        self.label_embeddings = self.vectorizer.encode(self.intent_labels)
    def extract(self, query: str):
        query_vec = self.vectorizer.encode(query)
        scores = torch.matmul(query_vec, self.label_embeddings.T)[0]
        probs = scores.softmax(dim=-1)
        result = dict(zip(self.intent_labels, probs.tolist()))
        # ----- derive usable filters -----
        filters = {
            "min_formality": 0.6 if result["formal office business setting"] > 0.4 else 0.0,
            "max_warmth": 0.6 if result["cold winter weather"] > 0.4 else 1.0,
            "min_comfort": 0.5 if result["comfortable relaxed clothing"] > 0.4 else 0.0,
            "sporty_ok": result["sport athletic activity"] > 0.4
        }
        return {
            "intent_scores": result,
            "filters": filters,
            "query_vec": query_vec
        }
```
### C Hybrid Filtering & Ranking
**Helper:** Context Match Score
```python
def context_match_score(item, filters, context):
    """
    Returns fuzzy context match score (0–1)
    """
    score = 1.0
    # Formality requirement
    if item["formality_score"] < filters["min_formality"]:
        return 0.0
    # Weather warmth constraint
    if item["weather_warmth"] > filters["max_warmth"]:
        return 0.0
    # Temperature context (if provided)
    if context and "temperature" in context:
        temp = context["temperature"]
        if temp < 10 and item["weather_warmth"] > 0.7:
            return 0.0
        if temp > 28 and item["weather_warmth"] < 0.3:
            return 0.0
    return score
```
**Hybrid Scoring Function**
```python
import torch.nn.functional as F
def hybrid_score(item, query_vec, filters, context, alpha=0.6, beta=0.4):
    """
    Combines context compatibility and semantic similarity.
    """
    context_score = context_match_score(item, filters, context)
    if context_score == 0:
        return 0.0
    item_vec = torch.tensor(item["embedding"]).unsqueeze(0)
    similarity = F.cosine_similarity(item_vec, query_vec).item()
    return alpha * context_score + beta * similarity
```
**Candidate Filtering & Ranking**
```python
def rank_items(items, query_vec, filters, context=None, top_k=20):
    """
    Filters and ranks wardrobe items.
    """
    scored = []
    for item in items:
        score = hybrid_score(item, query_vec, filters, context)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored[:top_k]]
```
**Example Usage:**
```python
vectorizer = QueryVectorizer()
intent_extractor = IntentExtractor(vectorizer)
query = "business meeting, comfortable"
result = intent_extractor.extract(query)
query_vec = result["query_vec"]
filters = result["filters"]
context = {
    "temperature": 15
}
ranked_items = rank_items(wardrobe_items, query_vec, filters, context)
print(filters)
print(ranked_items[:5])
```
## 3. Representation Layer (Vectorization)
### A. Visual Embeddings (Student CLIP)
**Visual Embedding Model:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
class StudentCLIPVisualEncoder(nn.Module):
    def __init__(self, embedding_dim=512, pretrained=True):
        super().__init__()
        # Load MobileNetV3 small/large backbone
        self.backbone = timm.create_model('mobilenetv3_small_100', pretrained=pretrained)
        
        # Replace classifier with a linear layer to embedding_dim
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, embedding_dim)
        
    def forward(self, x):
        # x: (B, 3, 224, 224)
        emb = self.backbone(x)  # (B, embedding_dim)
        emb = F.normalize(emb, dim=-1)  # Normalize for cosine similarity
        return emb
```
**Prepare the Fashionpedia Dataset:**
```python
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import json
class FashionpediaDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        with open(ann_file, 'r') as f:
            self.anns = json.load(f)['annotations']
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])
    
    def __len__(self):
        return len(self.anns)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.anns[idx]['image'])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = self.anns[idx]['category_id']  # optional, if needed for contrastive training
        return img, label
```
**Define Training Objective:**
```python
def distillation_loss(student_emb, teacher_emb):
    # teacher_emb already normalized
    return 2 - 2 * (student_emb * teacher_emb).sum(dim=-1).mean()
```
**Example Training Loop:**
```python
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize models
student = StudentCLIPVisualEncoder().to(device)
teacher = ...  # Pretrained CLIP ViT encoder, frozen
teacher.eval()
# Dataset & loader
dataset = FashionpediaDataset(img_dir='images/', ann_file='annotations.json')
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
# Optimizer
optimizer = Adam(student.parameters(), lr=1e-4)
# Training loop
for epoch in range(10):
    student.train()
    total_loss = 0
    for imgs, _ in tqdm(loader):
        imgs = imgs.to(device)
        with torch.no_grad():
            teacher_emb = teacher(imgs)  # (B, 512)
            teacher_emb = F.normalize(teacher_emb, dim=-1)
        
        student_emb = student(imgs)
        loss = distillation_loss(student_emb, teacher_emb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")
```
**Inference Example:**
```python
from PIL import Image
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])
img = Image.open("sample.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)
embedding = student(img_tensor)
print(embedding.shape)  # (1, 512)
```
## B. Text Embeddings
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
class TextEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", embedding_dim=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
    def forward(self, texts):
        # texts: list of strings
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(next(self.model.parameters()).device)
        outputs = self.model(**enc, output_hidden_states=True)
        # Mean pooling over token embeddings
        token_embeddings = outputs.last_hidden_state  # (B, seq_len, hidden)
        attention_mask = enc['attention_mask'].unsqueeze(-1)
        pooled = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(1)
        # Project to 512 dims if needed
        if pooled.shape[-1] != self.embedding_dim:
            pooled = nn.Linear(pooled.shape[-1], self.embedding_dim).to(pooled.device)(pooled)
        pooled = F.normalize(pooled, dim=-1)
        return pooled  # (B, 512)
```
**Usage:**
```python
text_encoder = TextEncoder().to(device)
texts = ["Red dress with floral pattern", "Black leather jacket"]
text_embeddings = text_encoder(texts)
print(text_embeddings.shape)  # (2, 512)
```
### C. Local Vector Store
```python
import sqlite3
import numpy as np
# Connect to local DB
conn = sqlite3.connect("vector_store.db")
c = conn.cursor()
# Create table
c.execute("""
CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    embedding BLOB,
    metadata TEXT
)
""")
conn.commit()
# Save embedding
def save_embedding(item_id, embedding, metadata=""):
    emb_bytes = embedding.cpu().numpy().tobytes()
    c.execute("INSERT OR REPLACE INTO items (item_id, embedding, metadata) VALUES (?, ?, ?)",
              (item_id, emb_bytes, metadata))
    conn.commit()
# Load all embeddings and search top_k by cosine similarity
def search(query_emb, top_k=5):
    c.execute("SELECT item_id, embedding, metadata FROM items")
    results = []
    for item_id, emb_bytes, metadata in c.fetchall():
        emb = torch.from_numpy(np.frombuffer(emb_bytes, dtype=np.float32))
        sim = F.cosine_similarity(query_emb, emb.unsqueeze(0))
        results.append((sim.item(), item_id, metadata))
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]
```
## 4. The “Brain” Layer (Recommendation Engine)
### A. Context-Aware Pre-Filter (Gatekeeper)
```python
import torch
import torch.nn.functional as F
class Gatekeeper:
    def __init__(self, wardrobe_db):
        """
        wardrobe_db: list of dictionaries containing item metadata and embeddings.
        """
        self.wardrobe = wardrobe_db
    def filter_and_rank(self, query_vec, filters, context, top_k=50):
        candidates = []
        for item in self.wardrobe:
            # 1. Hard Boolean Filters (Formality, Weather, Intent)
            if item["metrics"]["formality"] < filters.get("min_formality", 0.0):
                continue
            if item["metrics"]["weather_warmth"] > filters.get("max_warmth", 1.0):
                continue
                
            # Temperature context guardrail
            temp = context.get("temperature", 20)
            if temp < 10 and item["metrics"]["weather_warmth"] > 0.7:
                continue
            if temp > 28 and item["metrics"]["weather_warmth"] < 0.3:
                continue
            # 2. Semantic Similarity Score
            item_vec = torch.tensor(item["embedding"]).unsqueeze(0)
            similarity = F.cosine_similarity(item_vec, query_vec).item()
            candidates.append((similarity, item))
        # Sort strictly by semantic relevance to the query
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Return only the top_k candidate dictionaries
        return [item for score, item in candidates[:top_k]]
```
### B. Outfit Compatibility Model (Set-Transformer “Stylist”)
**The PyTorch Model**
```python
import torch
import torch.nn as nn
class OutfitSetTransformer(nn.Module):
    def __init__(self, embed_dim=512, num_categories=10, num_heads=4):
        super().__init__()
        
        # Category embedding to distinguish tops, bottoms, shoes, etc.
        self.category_embed = nn.Embedding(num_categories, embed_dim)
        
        # Self-attention layer to evaluate how items interact
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Feed-forward network for final scoring
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid() # Outputs compatibility score between 0 and 1
        )
    def forward(self, item_vectors, category_ids):
        # item_vectors: (Batch, N, 512)
        # category_ids: (Batch, N)
        
        # Add category information to visual/text embeddings
        cat_embs = self.category_embed(category_ids)
        x = item_vectors + cat_embs
        
        # Self-attention to find stylistic interactions
        attn_out, _ = self.attention(x, x, x)
        
        # Pooling: Aggregate the set into a single outfit vector (Mean Pooling)
        outfit_representation = attn_out.mean(dim=1) 
        
        # Score the outfit
        score = self.ffn(outfit_representation)
        return score
```
**Dataset Generation Logic**
```python
import random
import torch
from torch.utils.data import Dataset
class RobustOutfitDataset(Dataset):
    def __init__(self, fashionpedia_outfits, color_theory_func, co_occurrence_matrix, all_wardrobe_items):
        self.samples = []
        
        # 1. Positive Samples (Ground Truth)
        for outfit in fashionpedia_outfits:
            c_score = color_theory_func(outfit["colors"])
            final_score = 0.65 + (0.35 * c_score) # Max 1.0
            
            self.samples.append({
                "item_vectors": outfit["vectors"], 
                "category_ids": outfit["categories"],
                "target_score": final_score
            })
        # 2. Negative & Hard Negative Samples (Mix & Match)
        # Generate varied lengths (e.g., 3 to 6 items)
        for _ in range(len(fashionpedia_outfits) * 3): 
            num_items = random.randint(3, 6)
            mixed_outfit = random.sample(all_wardrobe_items, num_items)
            
            # Extract colors and calculate linear color score
            c_score = color_theory_func([item["color"] for item in mixed_outfit])
            
            # Calculate existence/co-occurrence score for this combination of categories
            combination_key = tuple(sorted([item["category"] for item in mixed_outfit]))
            e_score = co_occurrence_matrix.get(combination_key, 0.0)
            
            # Adjusted Math: Linear combination instead of squaring C
            # C contributes 60%, co-occurrence existence contributes 40%
            mixed_score = (0.6 * c_score) + (0.4 * e_score)
            
            # Keep outfits that score under 0.5 (includes both terrible <0.2 and mediocre 0.2-0.5)
            if mixed_score < 0.5:
                self.samples.append({
                    "item_vectors": [item["vector"] for item in mixed_outfit],
                    "category_ids": [item["category_id"] for item in mixed_outfit],
                    "target_score": mixed_score
                })
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Pad sequences here if using batching, or handle variable lengths in a custom collate_fn
        return torch.stack(sample["item_vectors"]), torch.tensor(sample["category_ids"]), torch.tensor([sample["target_score"]], dtype=torch.float32)
```
**Training Loop:**
```python
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
def train_stylist_model(model, dataset, epochs=10, batch_size=32, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Standard MSE for regression-like scoring
    criterion = nn.MSELoss() 
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Note: For variable length inputs in batches, you need a custom collate_fn to pad sequences.
    # For simplicity here, assuming batch_size=1 or padded tensors.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for item_vectors, category_ids, target_scores in progress_bar:
            item_vectors = item_vectors.to(device)
            category_ids = category_ids.to(device)
            target_scores = target_scores.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(item_vectors, category_ids)
            
            # Compute loss
            loss = criterion(predictions, target_scores)
            
            # Backward pass & optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(dataloader)
        print(f"End of Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")
    return model
```
### C. Final Ranking (Compatibility + Relevance)
```python
def generate_recommendations(query_vec, filters, context, wardrobe, bandit, top_n=5):
    # 1. Candidate Retrieval (Gatekeeper)
    # Narrows down 1000 items to 50 items that fit the weather/formality.
    gatekeeper = Gatekeeper(wardrobe)
    candidates = gatekeeper.filter_and_rank(query_vec, filters, context, top_k=50)
    
    # 2. Outfit Generation
    # Create valid combinations (e.g., Top + Bottom + Shoes + Accessory)
    outfit_combos = assemble_plausible_outfits(candidates)
    
    # 3. Final Scoring
    final_ranked_outfits = []
    alpha, beta = bandit.get_weights()
    
    for outfit in outfit_combos:
        # A. Stylist Score (Aesthetic Compatibility)
        comp_score = stylist_model(outfit.vectors, outfit.categories)
        
        # B. Relevance Score (Semantic matching to query)
        # Average the similarity of all items in the outfit to the query
        rel_score = torch.mean(torch.tensor([
            F.cosine_similarity(item.vector, query_vec) for item in outfit.items
        ])).item()
        
        # C. Weighted Fusion
        score = (alpha * comp_score) + (beta * rel_score)
        final_ranked_outfits.append((score, outfit, comp_score, rel_score))
        
    # Sort by final score
    final_ranked_outfits.sort(key=lambda x: x[0], reverse=True)
    return final_ranked_outfits[:top_n]
```
### D. Personalization (Reinforcement-lite / Bandit)
We treat the weights α (Compatibility) and β (Relevance) as parameters that shift based on user feedback (Like/Dislike).
**The Logic:** Gradient-Free Feedback Loop
Instead of complex backpropagation, we use a simple Exponentially Weighted Moving Average (EWMA). This ensures the app learns quickly at first but stabilizes over time.
```python
class PersonalPreferenceBandit:
    def __init__(self, initial_alpha=0.6, initial_beta=0.4, learning_rate=0.05):
        """
        alpha: Weight for Stylist Compatibility Score
        beta: Weight for Query Relevance Score
        """
        self.alpha = initial_alpha
        self.beta = initial_beta
        self.lr = learning_rate
    def update_weights(self, liked, compatibility_score, relevance_score):
        """
        liked: boolean (True if user liked/saved the outfit)
        """
        # If they liked a high-compatibility outfit but low-relevance one,
        # we nudge alpha up and beta down.
        
        reward = 1 if liked else -1
        
        # Calculate the impact of each component on the final decision
        # If compatibility was high and they liked it, strengthen alpha.
        # If it was high and they hated it, weaken alpha.
        delta_alpha = reward * (compatibility_score - 0.5) * self.lr
        delta_beta = reward * (relevance_score - 0.5) * self.lr
        
        self.alpha = max(0.1, min(0.9, self.alpha + delta_alpha))
        self.beta = max(0.1, min(0.9, self.beta + delta_beta))
        
        # Normalize weights so they always sum to 1.0
        total = self.alpha + self.beta
        self.alpha /= total
        self.beta /= total
    def get_weights(self):
        return self.alpha, self.beta
```