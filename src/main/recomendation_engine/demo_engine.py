import torch
import torch.nn.functional as F
from typing import List, Dict


# Assuming this is in your project
from src.main.recomendation_engine.model import OutfitEmbeddingTransformer
from src.config import STYLIST_BRAIN_DIR

class FashionBrain:
    def __init__(self, device: str = "cpu", mock_mode: bool = False):
        self.device = torch.device(device)
        self.mock_mode = mock_mode

        # --- Normal Initialization ---
        if not self.mock_mode:
            self.stylist = OutfitEmbeddingTransformer().to(self.device)
            self.stylist.load_state_dict(torch.load(STYLIST_BRAIN_DIR / "best_stylist.pth", map_location=self.device))
            self.stylist.eval()

        # User Centroid mapping their specific taste in the 128-dim hypersphere
        self.user_centroid = torch.randn(1, 128).to(self.device)
        self.user_centroid = F.normalize(self.user_centroid, p=2, dim=1)

        # Bandit Weights
        self.alpha = 0.6  # Weight for Aesthetic Style
        self.beta = 0.4  # Weight for Query Relevance
        self.eta = 0.15  # Learning rate

    def gatekeeper_filter(self, wardrobe_pool: List[Dict], query_vec: torch.Tensor, context: Dict, top_k: int = 40):
        """Filters the wardrobe based on context rules and semantic relevance."""
        candidates = []
        for garment in wardrobe_pool:
            # Extract nested attributes
            attrs = garment.get('attributes', {})
            warmth = attrs.get('weather_warmth', 0.5)
            formality = attrs.get('formality_score', 0.5)

            # Hard Boolean Rules
            if abs(warmth - context.get('temp_score', 0.5)) > 0.4: continue
            if abs(formality - context.get('formal_score', 0.5)) > 0.5: continue

            # Handle 2D tensor shape: tensor([[...]]) -> (1, 512)
            vec = garment['img_vec'].to(self.device)
            if vec.dim() == 1: vec = vec.unsqueeze(0)

            relevance = F.cosine_similarity(vec, query_vec, dim=1).item()
            candidates.append({'garment': garment, 'rel': relevance})

        candidates.sort(key=lambda x: x['rel'], reverse=True)
        return [c['garment'] for c in candidates[:top_k]]

    @torch.no_grad()
    def score_outfit(self, outfit_data: Dict, query_vec: torch.Tensor):
        """Scores a complete outfit dictionary against the User Centroid and Query."""
        # 1. Extract garments list from your dictionary structure
        items = outfit_data.get('garments', [])

        if not items:
            return 0.0, None, 0.0, 0.0

        # 2. Extract and format visual vectors
        # Your dict has 'img_vec' as tensor([[...]]). We squeeze to 1D, stack to (N, 512), then unsqueeze to (1, N, 512)
        item_vecs = torch.stack([i['img_vec'].squeeze(0) for i in items])
        item_vecs = item_vecs.unsqueeze(0).to(self.device)

        # 3. Extract Categories
        item_cats = torch.tensor([i['category_id'] for i in items]).unsqueeze(0).to(self.device)

        # 4. Generate Outfit Embedding (Stylist Forward Pass)
        if self.mock_mode:
            # MOCK 128-dim embedding for testing the math
            outfit_emb = torch.randn(1, 128).to(self.device)
            outfit_emb = F.normalize(outfit_emb, p=2, dim=1)
        else:
            outfit_emb = self.stylist(item_vecs, item_cats)

        # 5. Calculate Style Score (S_style) via Euclidean distance
        dist = torch.norm(outfit_emb - self.user_centroid, p=2).item()
        s_style = 1.0 / (1.0 + dist)

        # 6. Calculate Relevance Score (S_rel) via mean cosine similarity
        rel_scores = []
        for i in items:
            vec = i['img_vec'].to(self.device)
            if vec.dim() == 1: vec = vec.unsqueeze(0)
            rel_scores.append(F.cosine_similarity(vec, query_vec, dim=1).item())

        s_rel = sum(rel_scores) / len(items)

        # 7. Final Weighted Score
        final_score = (self.alpha * s_style) + (self.beta * s_rel)

        return final_score, outfit_emb, s_style, s_rel

    def update_feedback(self, outfit_emb: torch.Tensor, liked: bool, s_style: float, s_rel: float):
        """Updates User Centroid and Bandit weights based on feedback."""
        reward = 1.0 if liked else -1.0

        # Update Centroid
        if liked:
            self.user_centroid = (1 - self.eta) * self.user_centroid + (self.eta * outfit_emb)
        else:
            direction = outfit_emb - self.user_centroid
            self.user_centroid = self.user_centroid - (self.eta * 0.5 * direction)

        self.user_centroid = F.normalize(self.user_centroid, p=2, dim=1)

        # Update Bandit Weights
        self.alpha += reward * (s_style - 0.5) * 0.1
        self.beta += reward * (s_rel - 0.5) * 0.1

        # Clamp and normalize weights so alpha + beta = 1.0
        total = self.alpha + self.beta
        self.alpha = max(0.1, min(0.9, self.alpha / total))
        self.beta = max(0.1, min(0.9, self.beta / total))

        print(
            f"--> [Feedback applied] Liked: {liked} | New Weights - Alpha(Style): {self.alpha:.2f}, Beta(Rel): {self.beta:.2f}")


# =========================================================
# DEMO RUNNER
# =========================================================
if __name__ == "__main__":
    # Initialize in mock mode to bypass loading actual model weights
    brain = FashionBrain()

    # Simulate a user query vector (e.g., "comfortable casual date")
    mock_query_vec = torch.randn(1, 512)
    mock_query_vec = F.normalize(mock_query_vec, p=2, dim=1)

    # 1. Your Exact Input Payload
    demo_outfit = {
        'garments': [
            {
                'category_id': 23,
                'category_name': 'shoe',
                'confidence': 0.988,
                'img_vec': torch.randn(1, 512),  # Replaced raw tensor payload with random for execution
                'attributes': {
                    'weather_warmth': 0.508,
                    'formality_score': 0.496,
                }
            },
            {
                'category_id': 23,
                'category_name': 'shoe',  # (Assuming two shoes for the demo)
                'confidence': 0.992,
                'img_vec': torch.randn(1, 512),
                'attributes': {
                    'weather_warmth': 0.508,
                    'formality_score': 0.496,
                }
            }
        ]
    }

    print("\n--- Scoring Initial Outfit ---")
    score, emb, style_s, rel_s = brain.score_outfit(demo_outfit, mock_query_vec)
    print(f"Final Score: {score:.3f}")
    print(f"Style Sub-score: {style_s:.3f} | Relevance Sub-score: {rel_s:.3f}")

    print("\n--- Simulating User LIKE ---")
    # Simulate the user hitting the "Like" button
    brain.update_feedback(emb, liked=True, s_style=style_s, s_rel=rel_s)

    print("\n--- Scoring Same Outfit Post-Learning ---")
    # Because the user liked it, the centroid moved closer. The style score should be higher!
    score2, _, style_s2, rel_s2 = brain.score_outfit(demo_outfit, mock_query_vec)
    print(f"Final Score: {score2:.3f}")
    print(f"Style Sub-score: {style_s2:.3f} (Notice it increased!)")