# First, install the client: pip install qdrant-client
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Connect to your Docker container
client = QdrantClient("http://localhost:6333")

# # Create the collection
# client.create_collection(
#     collection_name="wardrobe_items",
#     vectors_config=VectorParams(size=512, distance=Distance.COSINE),
# )

print("✅ Qdrant Collection 'wardrobe_items' created successfully!")