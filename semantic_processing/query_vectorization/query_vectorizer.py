import torch
from transformers import DistilBertTokenizer

from semantic_processing.query_vectorization.model import DistilledQueryEncoder

# Assuming DistilledQueryEncoder is imported from the training script

device = "cuda" if torch.cuda.is_available() else "cpu"


class QueryVectorizer:
    """
    Encodes text queries into normalized 512-dim vectors
    using a custom distilled lightweight model.
    """

    def __init__(self, model_path="distilled_query_encoder.pth"):
        # Load the student tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        # Load the distilled student model
        self.model = DistilledQueryEncoder(out_dim=512).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def encode(self, query: str | list[str]):
        # Handle both single strings and lists (needed for intent extraction)
        if isinstance(query, str):
            query = [query]

        tokens = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            # Our custom model already handles the projection and normalization
            features = self.model(tokens['input_ids'], tokens['attention_mask'])

        return features

if __name__ == "__main__":
    # Example Usage
    vectorizer = QueryVectorizer()
    query_vec = vectorizer.encode("business meeting, looking for comfort")  #
    print(f"Output shape: {query_vec.shape}")  # Should be torch.Size([1, 512])
