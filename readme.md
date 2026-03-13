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

[//]: # (## 4. The “Brain” Layer &#40;Recommendation Engine&#41;)

[//]: # ()
[//]: # (This is where filtering + combination + scoring produces ranked outfits.)

[//]: # ()
[//]: # (### A. Context-Aware Pre-Filter &#40;Gatekeeper&#41;)

[//]: # ()
[//]: # (**Goal:** Reduce the wardrobe to a candidate set &#40;~50 items&#41; before trying combinations.)

[//]: # ()
[//]: # (#### Input:)

[//]: # ()
[//]: # (* Context &#40;occasion, weather, temperature, etc.&#41;)

[//]: # (* Query embedding)

[//]: # (* Wardrobe database)

[//]: # ()
[//]: # (#### Output:)

[//]: # ()
[//]: # (Candidate items list.)

[//]: # ()
[//]: # (#### Example:)

[//]: # ()
[//]: # (```txt)

[//]: # (Input:)

[//]: # (    Context: {"occasion":"office", "temp":15})

[//]: # (    Query: "business meeting")

[//]: # ()
[//]: # (Operation:)

[//]: # (    1&#41; filter formality_score >= 0.60)

[//]: # (    2&#41; filter weather_warmth <= threshold for cold)

[//]: # (    3&#41; rank by cosine similarity to query_vec, take top_k)

[//]: # ()
[//]: # (Output:)

[//]: # (    ["uuid_top_1", "uuid_bottom_5", "uuid_shoes_2", ...])

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (### B. Outfit Compatibility Model &#40;Set-Transformer “Stylist”&#41;)

[//]: # ()
[//]: # (**Model:** Small Set-Transformer &#40;attention-based&#41;)

[//]: # (**Purpose:** Given a set of item vectors &#40;+ category ids&#41;, output outfit compatibility score.)

[//]: # ()
[//]: # (#### Input:)

[//]: # ()
[//]: # (* item_vectors: &#40;1, N, 512&#41;)

[//]: # (* category_ids: &#40;1, N&#41;)

[//]: # ()
[//]: # (#### Output:)

[//]: # ()
[//]: # (compatibility_score &#40;0–1&#41;)

[//]: # ()
[//]: # (#### Example:)

[//]: # ()
[//]: # (```txt)

[//]: # (Input:)

[//]: # (    item_vectors: torch.Tensor&#40;&#41;  # &#40;1, 3, 512&#41; top/bottom/shoes)

[//]: # (    category_ids: torch.Tensor&#40;&#41;  # &#40;1, 3&#41; [0,1,2])

[//]: # ()
[//]: # (Output:)

[//]: # (    0.94)

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # (### C. Final Ranking &#40;Compatibility + Relevance&#41;)

[//]: # ()
[//]: # (Combine:)

[//]: # ()
[//]: # (* Aesthetic score &#40;transformer&#41;)

[//]: # (* Relevance score &#40;the mean of the score_item for each garment in the set&#41; // Refrence: 2C Hybrid Filtering  )

[//]: # ()
[//]: # (#### Example:)

[//]: # ()
[//]: # (```txt)

[//]: # (final_score = α * aesthetic_score + β * relevance_score)

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (### D. Personalization &#40;Reinforcement-lite / Bandit&#41;)

[//]: # ()
[//]: # (**Goal:** Adapt scoring weights &#40;α/β&#41; or last-layer parameters from feedback.)

[//]: # ()
[//]: # (#### Input:)

[//]: # ()
[//]: # (* user feedback &#40;like/dislike/skip&#41;)

[//]: # (* context + query embedding)

[//]: # ()
[//]: # (#### Output:)

[//]: # ()
[//]: # (Updated weights / preference parameters.)

[//]: # ()
[//]: # (#### Example:)

[//]: # ()
[//]: # (```txt)

[//]: # (Input:)

[//]: # (    reward = +1 &#40;liked&#41;)

[//]: # (    state = &#40;query_vec, context&#41;)

[//]: # ()
[//]: # (Output:)

[//]: # (    α := α + Δα)

[//]: # (    β := β + Δβ)

[//]: # (```)

[//]: # (---)

---

## 4. The “Brain” Layer (Recommendation Engine)

This is where filtering, set-embedding, and personalized spatial ranking produce the final recommended outfits.

### A. Context-Aware Pre-Filter (Gatekeeper)

**Goal:** Reduce the total wardrobe to a candidate set (~50 items) using hard constraints and semantic search.

#### Input:

* **Context**: Occasion, weather, temperature, etc. (from Attribute Classifier).
* **Query Embedding**: Vectorized user text (e.g., via Student-CLIP).
* **Wardrobe DB**: Pre-vectorized garment features and metadata.

#### Operation:

1. **Attribute Filter**: Remove items with mismatched `weather_warmth` or `formality_score` based on context.
2. **Semantic Retrieval**: Rank remaining items by cosine similarity between the garment vector and the query vector.
3. **Top-K Selection**: Pass the top 50 candidates to the combination generator.

---

### B. Outfit Compatibility Model (Set-Transformer “Stylist”)

**Model:** Triplet-based Set-Transformer.
**Purpose:** Map a set of items into a 128-dimensional **Style Space**. In this space, proximity represents stylistic compatibility.

#### Input:

* **item_vectors**: `(1, N, 512)` (Visual embeddings).
* **category_ids**: `(1, N)` (Categorical context).

#### Output:

* **outfit_embedding ($V_{outfit}$)**: A normalized `(1, 128)` vector representing the holistic "vibe" of the generated outfit.

---

### C. Final Ranking (Personalized Taste + Relevance)

Compatibility is no longer a static global score. It is dynamically calculated based on the outfit's distance to the user's specific **Taste Centroid** in the 128-dim Style Space.

#### Input:

* **$V_{outfit}$**: The 128-dim outfit embedding from the Stylist model.
* **$V_{user}$**: The user's personalized 128-dim profile centroid.
* **Query Similarity**: Cosine similarity of the items to the text query.

#### Scoring Logic:

1. **Personal Aesthetic Score ($S_{style}$)**: Calculated as $\frac{1}{1 + d}$, where $d$ is the Euclidean distance between $V_{outfit}$ and $V_{user}$.
2. **Relevance Score ($S_{rel}$)**: The mean cosine similarity between the query vector and each individual garment in the set.

#### Formula:

$$\text{final_score} = \alpha \cdot S_{style} + \beta \cdot S_{rel}$$

---

### D. Personalization (Taste Centroid + Bandit)

**Goal:** Adapt the user's location in the Style Space (*Taste Centroid*) AND their preference for strict matching vs. aesthetic exploration (*Bandit Weights*).

#### Input:

* **Feedback**: Reward ($+1$ for Like/Wear, $-1$ for Dislike/Skip).
* **$V_{outfit}$**: The embedding of the outfit they just interacted with.
* **State**: Current $V_{user}$ vector, $\alpha, \beta$ weights, and learning rate ($\gamma = 0.15$).

#### Operation 1: Centroid Update (Learning *What* They Like)

Move the user's profile vector through the embedding space based on interaction.

* **If Liked**: Pull user towards the outfit.

$$V_{user} := (1 - \gamma) \cdot V_{user} + (\gamma \cdot V_{outfit})$$


* **If Disliked**: Push user away from the outfit.

$$V_{user} := V_{user} - (\gamma \cdot 0.5 \cdot (V_{outfit} - V_{user}))$$


* *Note: Always L2-normalize $V_{user}$ after an update to keep it on the hypersphere.*

#### Operation 2: Bandit Update (Learning *How* They Shop)

Adjust the balance between Personal Style and Search Relevance.

* $\alpha := \alpha + \text{reward} \cdot (S_{style} - 0.5) \cdot 0.1$
* $\beta := \beta + \text{reward} \cdot (S_{rel} - 0.5) \cdot 0.1$
* *Note: Ensure $\alpha + \beta = 1.0$ and clamp values to $[0.1, 0.9]$.*

---

### Status Check:

* [x] **Architecture**: Shifted to Triplet-Loss Embedding.
* [x] **Dataset**: Implemented with hard-negative mining (Color Score thresholding).
* [ ] **Training**: Integrated with TensorBoard and Early Stopping.
* [ ] **Personalization**: Shifted from global heuristics to dynamic User Centroids.

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

This update to the **Final Pipeline Flow** integrates the transition from static scoring to the **Style Space Embedding** logic. It now reflects how individual garments are processed into a unified "vibe" vector that is then compared against a dynamic user profile.

---

## 7. Final Pipeline Flow

### A. Wardrobe Update (Offline/Ingestion)

1. **Detection & Segmentation**: Extract precise garment regions using RF-DETR.
2. **Vectorization**: Pass crops through `GarmentEmbedder` to get 512-dim visual features.
3. **Attribute & Color Extraction**:
* Predict `formality_score` and `weather_warmth`.
* Quantize colors for harmony checks.


4. **Local Indexing**: Store items in a local vector database with category and attribute metadata.

---

### B. User Query (The "Trigger")

1. **Semantic Encoding**: Convert text (e.g., "Dinner date in Paris") into a 512-dim query embedding using the Student-CLIP model.
2. **Intent Parsing**: Identify required attributes (e.g., "Date" → High Formality, "Cold" → High Warmth).

---

### C. Candidate Filtering (Gatekeeper)

1. **Contextual Pruning**: Drop items that violate weather/formality constraints.
2. **Semantic Ranking**: Score remaining items by cosine similarity to the query embedding.
3. **Top-K Selection**: Keep the top ~40 items to prevent combinatorial explosion in the next step.

---

### D. Outfit Generation (Combinatorics)

1. **Set Assembly**: Group filtered items into valid permutations (e.g., 1 Top + 1 Bottom + 1 Outerwear + 1 Shoes).
2. **Structure Check**: Ensure category diversity (preventing an outfit of 3 shirts).

---

### E. Scoring & Ranking (The Brain)

1. **Personalized Style Mapping**: Pass the set through the **Triplet Set-Transformer** to project the outfit into the 128-dim Style Space.
2. **Distance Calculation**: Measure the Euclidean distance between the outfit vector ($V_{outfit}$) and the **User Taste Centroid** ($V_{user}$).
3. **Hybrid Scoring**:
* $S_{style} = \frac{1}{1 + \text{dist}}$
* $S_{rel} = \text{mean similarity to query}$
* $\text{Final} = \alpha \cdot S_{style} + \beta \cdot S_{rel}$



---

### F. Learning Loop (The Feedback)

1. **Representation Update**: On a "Like" or "Wear" action, shift the **User Taste Centroid** toward the outfit's coordinates in the Style Space.
2. **Behavioral Update**: Adjust the Bandit weights ($\alpha, \beta$) based on whether the user prioritized the "aesthetic vibe" or the "search relevance" of the result.
