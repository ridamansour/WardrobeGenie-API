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

#### Status:
- [x] Modeled
- [x] Tested
- [x] Trained
- [ ] Tested

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

#### Status:
- [x] Modeled
- [x] Tested
- [x] Trained
- [x] Tested
- [ ] Converted to TFLite

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
#### Status:
- [x] Developed
- [x] Tested

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

#### Status:
- [x] Developed
- [x] Tested

---
### E. Color Harmony Score (Function)

#### Input:

Cropped clothing item images and the number of dominant colors per garment.

#### Output:

Color harmony score between 0 and 1 of the cropped images combined

#### Example:
```txt
Input:
    images : list[PIL.Image, PIL.Image, PIL.Image] # 3 images
    k : 5 # number of dominant colors per garment

Output:
    0.75 #  The color harmony score is 75%
```

#### Status:
- [x] Developed
- [x] Tested

---

[//]: # (### F. On-Device Implementation &#40;Mobile Deployment&#41;)

[//]: # ()
[//]: # (**Goal:** Run inference locally for privacy + offline operation.)

[//]: # ()
[//]: # (* Convert models to **TFLite** &#40;Android&#41; or **CoreML** &#40;iOS/macOS&#41;.)

[//]: # (* Apply **quantization** to reduce size and accelerate inference.)

[//]: # ()
[//]: # (**Note:** Inference time may be several seconds depending on device/model size.)

[//]: # ()
[//]: # (#### Status:)

[//]: # (* **TFLite:** Haven’t started yet.)

[//]: # (* **CoreML:** Haven’t started yet.)

[//]: # (---)

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

#### Status:
- [x] Modeled
- [x] Tested
- [x] Trained
- [x] Tested

[//]: # (- [ ] Converted to TFLite)

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
#### Status:
- [x] Developed
- [ ] Tested

---

### C. Hybrid Filtering (Semantic + Boolean)

**Idea:** A combined score:

* semantic similarity (cosine)
* context constraints (weather/formality rules)

#### Example:

```txt
score_item = α * context_score + β * cosine(item_vec, query_vec)
```

#### Status:
- [x] Developed
- [ ] Tested

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

#### Status:
- [x] Modeled
- [x] Tested
- [x] Trained
- [ ] Tested
- [ ] Converted to TFLite

---

### B. Text Embeddings

**Purpose:** Ensure text and image vectors live in the same embedding space (512-dim).

#### Status:
- [x] Done
- [x] Tested

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

#### Status:
- [x] Developed
- [ ] Tested

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

Status:
- [ ] Developed
- [ ] Tested

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

#### Status:
- [ ] Modeled
- [ ] Tested
- [ ] Trained
- [ ] Tested
- [ ] Converted to TFLite

---

### C. Final Ranking (Compatibility + Relevance)

Combine:

* Aesthetic score (transformer)
* Relevance score (the mean of the score_item for each garment in the set) // Refrence: 2C Hybrid Filtering  

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

#### Status:
- [ ] Modeled
- [ ] Tested
- [ ] Trained
- [ ] Tested
- [ ] Converted to TFLite

---

## 5. Training Strategy & Data

### A. Cloud Training

All heavy training is done on device (GPU):

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