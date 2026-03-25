# WardrobeGenie: Intelligent Fashion Recommendation Engine

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-009688.svg?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-FF5252.svg)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker)
![Apache Airflow](https://img.shields.io/badge/Apache_Airflow-2.8+-017CEE.svg?logo=apache-airflow)

WardrobeGenie is a context-aware, machine learning-powered digital stylist. It transcends standard static rule-based engines by utilizing a **Retrieval-Augmented Generation (RAG)** architecture for computer vision, continuously learning from user feedback to map an individual's unique aesthetic.

## The Product: What the API Does

From a consumer standpoint, the WardrobeGenie API drives a three-step loop:

1. **The Digital Closet (Ingestion):** Users upload photos of their clothes. The API automatically detects the garments in the image, and assigns semantic attributes (formality, weather-warmth, color harmony) to each garment.
2. **The Stylist (Recommendation):** Users send natural language queries (e.g., *"Business casual for a rainy day"*). The API translates this text into mathematical vectors, filters the wardrobe using Qdrant, and generates a mathematically optimized combinatorial outfit (Top + Bottom + Shoes).
3. **The Vote (Personalization):** Users vote on the recommended outfits (Like, Dislike, or Skip). This feedback triggers a Reinforcement Learning loop, shifting the user's "style centroid" in the vector space so future recommendations adapt to their exact taste.

---

## Enterprise Architecture & Separation of Concerns

To ensure production-grade reliability, the infrastructure is strictly decoupled into two isolated Dockerized stacks:

### 1. Online Serving Stack (Low Latency API)
Managed via `docker-compose.yml`, this stack handles real-time mobile app requests.
* **FastAPI:** Asynchronous API gateway.
* **Qdrant Vector Engine:** Hardware-accelerated Approximate Nearest Neighbor (ANN) search over 512-dimensional garment embeddings.
* **The "Brain" (Set-Transformer):** Dynamically scores combinatorial outfits based on context.

### 2. Offline Orchestration Stack (High Throughput ETL)
Managed via `docker-compose-airflow.yml`, this stack safely processes heavy data without bottlenecking the live API.
* **Apache Airflow:** Orchestrates DAGs for batch image ingestion, feature extraction, and continuous model training.
* **CI/CD Pipeline:** GitHub Actions automatically tests API routing and logic via mocked PyTorch models on every push to `main`.

---

## MLOps & Airflow Pipelines

The true engine behind WardrobeGenie is its automated MLOps lifecycle, orchestrated entirely by **Apache Airflow**. The system utilizes three primary Directed Acyclic Graphs (DAGs) to maintain the ML models:

1. **`wardrobegenie_batch_ingestion`**
   * Periodically fetches bulk user uploads from cloud storage (e.g., AWS S3).
   * Validates data idempotency and routes images through the offline Perception Layer (YOLOS cropping and Attribute Extraction).
   * Generates 512-dim visual embeddings and pushes them directly into the Qdrant database.
2. **`wardrobegenie_data_synthesis`**
   * Automates the formatting of COCO datasets for vision tasks.
   * Programmatically synthesizes massive datasets of realistic NLP queries—utilizing combinatorial templates, simulated mobile "fat-finger" typos, and regional spelling variants—to train and robustly refine the semantic intent models used by the Gatekeeper.
3. **`wardrobegenie_continuous_training`**
   * The reinforcement loop. Aggregates the "Vote" feedback from users.
   * Retrains the Triplet Set-Transformer in the background on a scheduled interval.
   * Validates model accuracy metrics and seamlessly updates the `/models` directory without API downtime.

---

## The Machine Learning Pipeline

WardrobeGenie operates on a four-tier architecture, transforming raw user uploads and text queries into dynamic, personalized outfit recommendations through a continuous feedback loop.

### 1. Perception Layer (Image Processing)
* **Detection & Segmentation:** Isolates individual garments from raw user uploads. 
  * *Note: The primary **RF-DETR Nano** model is currently fine-tuning in the Airflow pipeline. The live API temporarily utilizes a robust **YOLO-based** detector as a fallback during this training phase.*
* **Multi-Attribute Classifier:** An EfficientNet-B0 multi-head model predicts fit, style, weather appropriateness (warmth), and formality in a single forward pass.
* **Color Quantizer:** K-Means clustering extracts dominant HEX codes from segmented items to compute downstream color harmony scores.

### 2. Semantic & Representation Layers
* **Query Vectorization:** A Distilled BERT / MobileCLIP text branch encodes natural language user queries (e.g., "casual dinner in autumn") into dense 512-dimensional vectors.
* **Visual Embeddings:** A MobileNetV3 student model (distilled from a CLIP ViT teacher) converts cropped garment images into normalized 512-dimensional "vibe" embeddings for rapid similarity search.

### 3. The "Brain" Layer (Recommendation Engine)
* **Hybrid Gatekeeper:** A first-pass filter that combines semantic similarity (cosine distance between text and image embeddings) with hard boolean context constraints (e.g., `formality_score >= 0.60` and `weather == cold`).
* **Set-Transformer (The Stylist):** An attention-based neural network that evaluates a candidate set of items (e.g., Top, Bottom, Shoes) simultaneously, outputting a unified aesthetic compatibility score for the entire outfit.

### 4. The Adaptive Layer (Online Preference Learning)
* **Dynamic Taste Tracking:** Users are represented by a 128-dim "taste centroid." Positive feedback (saves/likes) uses an Exponential Moving Average (EMA) to pull this centroid toward the outfit's embedding, while skips push it away.
* **Adaptive Weighting:** The final recommendation balances Aesthetic Style (α) and Query Relevance (β). These weights shift in real-time based on instantaneous user feedback, automatically prioritizing whichever metric the user actively engages with most.

---

## Core API Endpoints

* **`POST /analyze-outfit`**
  * Accepts raw image uploads. Returns bounding boxes, quantized colors, and predicted attributes.
* **`POST /recommend/search`**
  * **The RAG Endpoint.** Accepts a text query and user context. Queries the Qdrant database for Top-K candidates, passes them through the Gatekeeper, and returns full outfits.
* **`POST /feedback`**
  * **The Vote.** Accepts boolean user feedback to dynamically update the $\alpha$ and $\beta$ scoring weights and adjust the user's stylistic centroid.

---

## Local Deployment (Quickstart)

### 1. Boot the Serving Stack
Boot the FastAPI backend and Qdrant Vector Database. 
```bash
docker-compose -f docker-compose-api.yml up -d
```

### 2. Seed the Vector Database
Populate Qdrant with simulated 512-dim garment data to test the RAG pipeline.
```bash
pip install -r requirements.txt
python seed_qdrant.py
```

### 3. Access the Interactive API
Navigate to **`http://localhost:8000/docs`** to view the auto-generated Swagger UI and test live image uploads and vector search recommendations.

### 4. Boot the MLOps Pipeline (Airflow)
To view and trigger the offline data ingestion, feature extraction, and model training DAGs, initialize the Airflow stack. 
*(Note: Set your local user ID first so Docker has the correct file permissions).*
```bash
echo -e "AIRFLOW_UID=$(id -u)" > .env
docker-compose -f docker-compose-airflow.yml up -d
```
Navigate to **`http://localhost:8080`** to access the Airflow UI. 
* **Username:** `airflow`
* **Password:** `airflow`