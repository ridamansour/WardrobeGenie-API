"""
main.py
WardrobeGenie Online Serving API
================================
Provides real-time endpoints for the mobile app to analyze uploads,
generate combinatorial recommendations, and process reinforcement feedback.
"""

import io
import os
import torch
import numpy as np
import itertools
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
from uuid import uuid4
from PIL import Image

from qdrant_client import QdrantClient

# --- Local Module Imports ---
from perception_layer.clothing_detection_yolo_backup.inference import GarmentDetector
from perception_layer.multi_attribute_classifier.inference import AttributePredictor
from representation_layer.visual_embeddings.inference import GarmentEmbedder
from recomendation_engine.brain_engine import FashionBrain
from perception_layer.color_utils import quantize_colors, harmony_score_from_images
from semantic_processing.query_vectorization.query_vectorizer import QueryVectorizer
from semantic_processing.intent_filter_extractor.IntentExtractor import IntentExtractor

# --- Configuration & Paths ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "public/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# ---------------------------------------------------------
# 1. Global State & App Lifespan (Memory Safe Loading)
# ---------------------------------------------------------
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing WardrobeGenie ML Backend...")

    # Base path for all models
    MODELS_DIR = BASE_DIR / "models"
    print(f"Models will be loaded from {MODELS_DIR}")
    if not MODELS_DIR.exists():
        print(f"{Path(__file__).resolve()=}")
        print(f"{Path(__file__).resolve().parent=}")
        print(f"{Path(__file__).resolve().parent.parent=}")
        raise FileNotFoundError(f"Model directory not found: {MODELS_DIR}")

    # 1. Perception
    print("Loading YOLOS Detector...")
    YOLO_MODEL_PATH = MODELS_DIR / "yolos-fashionpedia/"
    ml_models['detector'] = GarmentDetector(str(YOLO_MODEL_PATH))


    ATTR_MODEL_PATH = MODELS_DIR / "attribute_predictor" / "1b" / "best_model.pt"
    print(f"Loading Attribute Predictor in {ATTR_MODEL_PATH} with {DEVICE=}")
    ml_models['attribute_predictor'] = AttributePredictor(str(ATTR_MODEL_PATH), device=DEVICE)

    # 2. Representation
    print("Loading Garment Embedder...")
    EMBED_MODEL_PATH = MODELS_DIR / "visual_embedder" / "best_student_model.pth"
    ml_models['garment_embedder'] = GarmentEmbedder(str(EMBED_MODEL_PATH), device=DEVICE)

    # 3. Semantic
    print("Loading NLP Models...")
    NLP_MODEL_PATH = MODELS_DIR / "nlp_query" / "distilled_query_encoder.pth"
    vectorizer = QueryVectorizer(model_path=str(NLP_MODEL_PATH))
    ml_models['intent_extractor'] = IntentExtractor(vectorizer=vectorizer)

    # 4. The Brain
    print("Loading Set-Transformer...")
    STYLIST_MODEL_PATH = MODELS_DIR / "stylist_brain" / "best_stylist.pth"
    ml_models['brain'] = FashionBrain(model_path=str(STYLIST_MODEL_PATH), wardrobe_path=None, device=DEVICE)

    # 5. Vector Database
    print("Connecting to Qdrant Vector Engine...")
    qdrant_url = os.getenv("VECTOR_DB_URL", "http://localhost:6333")
    ml_models['qdrant'] = QdrantClient(qdrant_url)

    print("All models loaded into VRAM. API is live!")
    yield

    print("Shutting down... Clearing VRAM.")
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.mps.is_available():
        torch.mps.empty_cache()


app = FastAPI(title="WardrobeGenie ML Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 2. Constants & Mappings
# ---------------------------------------------------------
CATEGORY_MAP = {
    'shirt, blouse': 'top', 'top, t-shirt, sweatshirt': 'top', 'sweater': 'top',
    'cardigan': 'top', 'dress': 'top', 'jumpsuit': 'top', 'pants': 'bottom',
    'shorts': 'bottom', 'skirt': 'bottom', 'jacket': 'outerwear', 'vest': 'outerwear',
    'coat': 'outerwear', 'cape': 'outerwear', 'shoe': 'shoes', 'glasses': 'accessories',
    'hat': 'accessories', 'headband, head covering, hair accessory': 'accessories',
    'tie': 'accessories', 'glove': 'accessories', 'watch': 'accessories',
    'belt': 'accessories', 'bag, wallet': 'accessories', 'scarf': 'accessories',
    'umbrella': 'accessories',
}

STYLE_RULES = {
    ("Formal", "Elegant"): {"formality": 0.9, "tips": ["Focus on clean lines and high-quality materials.",
                                                       "Stick to a refined color palette."]},
    ("Work", "Minimal"): {"formality": 0.7, "tips": ["Less is more. Avoid distracting patterns.",
                                                     "Neutral tones create a professional look."]},
    ("Casual", "Relaxed"): {"formality": 0.2, "tips": ["Prioritize comfort without sacrificing fit.",
                                                       "Loose but intentional layering works well."]},
    ("Date Night", "Bold"): {"formality": 0.65, "tips": ["Express your personality with a standout color or pattern."]},
    ("University", "Minimal"): {"formality": 0.3,
                                "tips": ["Simple, clean essentials that transition from lecture to social."]}
}


# ---------------------------------------------------------
# 3. Pydantic Schemas
# ---------------------------------------------------------
class Color(BaseModel):
    hex: str
    percentage: float


class OutfitItem(BaseModel):
    clothing_type: str
    category_id: int
    category_name: Optional[str] = None
    material: str
    formality_score: float
    weather_warmth: float
    colors: List[Color]
    image_path: str
    visual_embedding: Optional[List[float]] = None
    confidence: Optional[float] = None


class CurrentOutfitRepresentation(BaseModel):
    items: List[OutfitItem]
    source_image_path: Optional[str] = None
    outfit_embedding: Optional[List[float]] = None
    s_style: Optional[float] = None
    s_rel: Optional[float] = None


class ContextInput(BaseModel):
    occasion: str
    style_intent: str
    weather: Optional[str] = None
    user_centroid: Optional[List[float]] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None


class Recommendation(BaseModel):
    items: List[OutfitItem]
    compatibility_score: float
    color_harmony: Optional[float] = None
    formality_match: Optional[float] = None
    weather_compat: Optional[float] = None
    style_tips: Optional[List[str]] = None
    explanation: Optional[str] = None
    intents: Optional[dict] = None
    filters: Optional[dict] = None
    outfit_embedding: Optional[List[float]] = None
    s_style: Optional[float] = None
    s_rel: Optional[float] = None


class RecommendFullRequest(BaseModel):
    context: ContextInput
    outfit: CurrentOutfitRepresentation


class RecommendFullResponse(BaseModel):
    recommendations: List[Recommendation]


class RecommendSearchRequest(BaseModel):
    context: ContextInput
    query: str  # e.g., "elegant formal outfit for a winter wedding"


class FeedbackRequest(BaseModel):
    session_id: str
    outfit_embedding: List[float]
    liked: bool
    s_style: float
    s_rel: float
    user_centroid: Optional[List[float]] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None


# ---------------------------------------------------------
# 4. Helper Functions
# ---------------------------------------------------------
def calculate_color_harmony(images: List[Image.Image]) -> float:
    if not images: return 0.0
    return harmony_score_from_images(images)


def generate_valid_outfits(candidates: List[Dict], max_combinations=10):
    """Combinatorial logic preventing absurd pairings (e.g. Dresses + Pants)."""
    tops = [c for c in candidates if c['orig_item'].clothing_type == 'top']
    bottoms = [c for c in candidates if c['orig_item'].clothing_type == 'bottom']
    shoes = [c for c in candidates if c['orig_item'].clothing_type == 'shoes']
    outerwear = [c for c in candidates if c['orig_item'].clothing_type == 'outerwear']

    one_pieces = [t for t in tops if t['orig_item'].category_name in ['dress', 'jumpsuit']]
    standard_tops = [t for t in tops if t not in one_pieces]

    outfit_candidates = []

    # Two-Piece Outfits
    if standard_tops and bottoms and shoes:
        for t, b, s in itertools.islice(itertools.product(standard_tops[:5], bottoms[:5], shoes[:5]), max_combinations):
            outfit = [t, b, s]
            if outerwear and np.random.rand() > 0.5:
                outfit.append(outerwear[0])
            outfit_candidates.append(outfit)

    # One-Piece Outfits
    if one_pieces and shoes:
        for op, s in itertools.islice(itertools.product(one_pieces[:5], shoes[:5]), max_combinations):
            outfit_candidates.append([op, s])

    # Fallback if no shoes are in the pool
    if not shoes and standard_tops and bottoms:
        for t, b in itertools.islice(itertools.product(standard_tops[:5], bottoms[:5]), max_combinations):
            outfit_candidates.append([t, b])

    return outfit_candidates


# ---------------------------------------------------------
# 5. API Endpoints
# ---------------------------------------------------------
@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "models_loaded": "brain" in ml_models}


@app.post("/analyze-outfit", response_model=CurrentOutfitRepresentation, tags=["Perception"])
async def analyze_outfit(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Must be an image")

    detector: GarmentDetector = ml_models['detector']
    attribute_predictor: AttributePredictor = ml_models['attribute_predictor']
    garment_embedder: GarmentEmbedder = ml_models['garment_embedder']
    brain: FashionBrain = ml_models['brain']

    session_id = uuid4().hex
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    content = await image.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")

    detections = detector.predict(img, threshold=0.5)
    items = []

    for idx, det in enumerate(detections):
        cat_id = det.get("category_id", 0)
        cat_name = det["category_name"]

        mapped_type = CATEGORY_MAP.get(cat_name, cat_name)
        if cat_name in ['dress', 'jumpsuit']: mapped_type = 'top'

        cropped_img = det["image"]
        attr_results = attribute_predictor.predict(cropped_img, category_id=cat_id)
        visual_emb = garment_embedder.embed_crop(cropped_img).squeeze().tolist()

        color_data = quantize_colors(cropped_img, k=3)
        colors = [Color(hex=c[0], percentage=c[1]) for c in color_data]

        crop_path = f"{session_id}/crop_{idx}.jpg"
        cropped_img.save(UPLOAD_DIR / crop_path)

        items.append(OutfitItem(
            clothing_type=mapped_type, category_id=cat_id, category_name=cat_name,
            material="unknown", formality_score=attr_results["formality_score"],
            weather_warmth=attr_results["weather_warmth"], colors=colors,
            image_path=crop_path, visual_embedding=visual_emb, confidence=det["confidence"]
        ))

    outfit_embedding, s_style, s_rel = None, 0.5, 0.5
    if items:
        item_vecs = torch.stack([torch.tensor(i.visual_embedding) for i in items if i.visual_embedding]).unsqueeze(
            0).to(DEVICE)
        item_cats = torch.tensor([i.category_id for i in items if i.visual_embedding]).unsqueeze(0).to(DEVICE)

        if item_vecs.shape[1] > 0:
            with torch.no_grad():
                outfit_emb_tensor = brain.stylist(item_vecs, item_cats)
                outfit_embedding = outfit_emb_tensor.squeeze().tolist()
                dist = torch.norm(outfit_emb_tensor - brain.user_centroid, p=2).item()
                s_style = 1.0 / (1.0 + dist)

    return CurrentOutfitRepresentation(items=items, outfit_embedding=outfit_embedding, s_style=round(s_style, 2),
                                       s_rel=s_rel)


@app.post("/recommend/search", response_model=RecommendFullResponse, tags=["Recommendation"])
def recommend_from_qdrant(payload: RecommendSearchRequest):
    qdrant: QdrantClient = ml_models['qdrant']
    intent_extractor: IntentExtractor = ml_models['intent_extractor']

    brain: FashionBrain = ml_models['brain']

    # 1. Convert text "date night" into a 512-dim mathematical vector
    q_vec = intent_extractor.vectorizer.vectorize(payload.query)
    q_vec_list = q_vec.squeeze().tolist()

    # FIX 2: Create the PyTorch tensor variant for the brain to score
    q_vec_tensor = torch.tensor(q_vec).to(DEVICE)

    # 2. Ask Qdrant for the top 100 closest visual matches in milliseconds
    search_results = qdrant.search(
        collection_name="wardrobe_items",
        query_vector=q_vec_list,
        limit=100,
        with_payload=True,
        with_vectors=True
    )

    # 3. Format Database Results for the Gatekeeper & Brain
    pool_items = []
    for hit in search_results:
        # Reconstruct the OutfitItem from the database payload
        item = OutfitItem(
            clothing_type=hit.payload.get("clothing_type"),
            category_id=hit.payload.get("category_id", 0),
            material="unknown",
            formality_score=hit.payload.get("formality_score", 0.5),
            weather_warmth=hit.payload.get("weather_warmth", 0.5),
            colors=hit.payload.get("colors", []),
            image_path=hit.payload.get("image_path", ""),
            visual_embedding=hit.vector
        )

        pool_items.append({
            'vec': torch.tensor(hit.vector).to(DEVICE),
            'cat': item.category_id,
            'warmth': item.weather_warmth,
            'formality': item.formality_score,
            'orig_item': item
        })

    # 4. Generate Combinatorial Outfits & Score them
    outfit_candidates = generate_valid_outfits(pool_items, max_combinations=10)
    recommendations = []

    for o_items in outfit_candidates:
        comp_score, outfit_emb_tensor = brain.score_outfit(o_items, q_vec_tensor)

        recommendations.append(Recommendation(
            items=[i['orig_item'] for i in o_items],
            compatibility_score=round(comp_score, 2),
            style_tips=["RAG Vector Search match."],
            outfit_embedding=outfit_emb_tensor.squeeze().tolist()
        ))

    recommendations.sort(key=lambda x: x.compatibility_score, reverse=True)
    return RecommendFullResponse(recommendations=recommendations[:4])


@app.post("/feedback", tags=["Reinforcement Learning"])
def update_feedback(payload: FeedbackRequest, background_tasks: BackgroundTasks):
    brain: FashionBrain = ml_models['brain']

    if payload.user_centroid: brain.user_centroid = torch.tensor([payload.user_centroid]).to(DEVICE)
    if payload.alpha: brain.alpha = payload.alpha
    if payload.beta: brain.beta = payload.beta

    outfit_emb = torch.tensor([payload.outfit_embedding]).to(DEVICE)

    # Process the heavy math in the background
    background_tasks.add_task(brain.update_feedback, outfit_emb, payload.liked, payload.s_style, payload.s_rel)

    return {
        "status": "Centroid shifting in background...",
        "new_alpha": brain.alpha,
        "new_beta": brain.beta
    }