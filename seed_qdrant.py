"""
seed_qdrant.py
Populates the Qdrant Vector DB with 100 realistic mock garments.
Run this once to prep your database for the Mälardalen University demo!
"""
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Connect to Qdrant (running on your Mac via Docker port 6333)
client = QdrantClient("http://localhost:6333")

COLLECTION_NAME = "wardrobe_items"

# 1. Initialize the Collection (512 dimensions for your MobileNet Student)

# client.create_collection(
#     collection_name=COLLECTION_NAME,
#     vectors_config=VectorParams(size=512, distance=Distance.COSINE),
# )

print("🌱 Seeding Qdrant with 100 garments...")

categories = ['top', 'bottom', 'shoes', 'outerwear']
points = []

for i in range(100):
    # Create a realistic 512-dim embedding (normalized)
    vector = np.random.rand(512).astype(np.float32)
    vector = vector / np.linalg.norm(vector)

    # Generate metadata payload
    cat = np.random.choice(categories)
    payload = {
        "clothing_type": cat,
        "category_id": categories.index(cat),
        "formality_score": round(np.random.uniform(0.1, 0.9), 2),
        "weather_warmth": round(np.random.uniform(0.1, 0.9), 2),
        "image_path": f"mock_uploads/item_{i}.jpg",
        "colors": [{"hex": "#000000", "percentage": 100.0}]
    }

    points.append(PointStruct(id=str(uuid.uuid4()), vector=vector.tolist(), payload=payload))

# Upload in a single batch
client.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"Successfully injected {len(points)} items into Qdrant!")
print("Go check your dashboard at http://localhost:6333/dashboard")