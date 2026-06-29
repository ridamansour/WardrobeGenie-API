import torch
import torch.nn.functional as F

class IntentExtractor:
    """
    Zero-shot intent detection using text embedding similarity.
    Converts soft query meaning into hard filtering constraints.
    """

    def __init__(self, vectorizer):
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

        # Precompute label embeddings once
        with torch.no_grad():
            self.label_embeddings = self.vectorizer.encode(self.intent_labels)

    def extract(self, query: str):
        query_vec = self.vectorizer.encode(query)

        # cosine similarity (dot product since normalized)
        scores = torch.matmul(query_vec, self.label_embeddings.T)[0]

        probs = F.softmax(scores, dim=-1)

        intent_scores = dict(zip(self.intent_labels, probs.tolist()))

        # ----- derive usable filters -----
        filters = {
            # enforce formality if detected
            "min_formality": 0.6 if intent_scores["formal office business setting"] > 0.40 else 0.0,

            # restrict heavy winter items if not cold
            "max_warmth": 0.6 if intent_scores["cold winter weather"] > 0.40 else 1.0,

            # require comfort if requested
            "min_comfort": 0.5 if intent_scores["comfortable relaxed clothing"] > 0.40 else 0.0,

            # allow sporty items
            "sporty_ok": intent_scores["sport athletic activity"] > 0.40,
        }

        return {
            "intent_scores": intent_scores,
            "filters": filters,
            "query_vec": query_vec
        }