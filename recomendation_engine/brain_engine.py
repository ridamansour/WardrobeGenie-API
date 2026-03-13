import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from model import OutfitEmbeddingTransformer  # Your Triplet Transformer
from perception_layer import color_utils


class FashionBrain:
    def __init__(self, model_path: str, wardrobe_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        # 1. Load the Stylist (The Brain)
        self.stylist = OutfitEmbeddingTransformer().to(self.device)
        self.stylist.load_state_dict(torch.load(model_path, map_location=self.device))
        self.stylist.eval()

        # 2. Load Wardrobe & User Profile
        # Expected wardrobe format: [{'id': 1, 'vec': tensor, 'cat': 1, 'warmth': 0.5, 'formality': 0.8}, ...]
        data = torch.load(wardrobe_path)
        self.wardrobe = data['items']

        # User Centroid: 128-dim vector representing the user's "vibe"
        # If new user, initialize with the global mean (saved during training)
        self.user_centroid = data.get('user_centroid', torch.randn(1, 128)).to(self.device)

        # Bandit Weights: α (Style) vs β (Query Relevance)
        self.alpha = 0.6
        self.beta = 0.4
        self.eta = 0.05  # Learning rate for feedback

    def gatekeeper_filter(self, query_vec: torch.Tensor, context: Dict[str, float], top_k: int = 40):
        """Phase 1: Filter by context (Weather/Formality) and Semantic Similarity."""
        candidates = []

        for item in self.wardrobe:
            # Hard filters for weather and formality (from Attribute Classifier)
            if abs(item['warmth'] - context.get('temp_score', 0.5)) > 0.4:
                continue
            if abs(item['formality'] - context.get('formal_score', 0.5)) > 0.5:
                continue

            # Relevance: How well does this item match the user's specific text query?
            relevance = F.cosine_similarity(item['vec'].unsqueeze(0), query_vec, dim=1).item()
            candidates.append({'item': item, 'rel': relevance})

        # Sort by relevance and take top_k to reduce combination explosion
        candidates.sort(key=lambda x: x['rel'], reverse=True)
        return [c['item'] for c in candidates[:top_k]]

    @torch.no_grad()
    def score_outfit(self, items: List[Dict], query_vec: torch.Tensor):
        """Phase 2: Use the Stylist to calculate Personal Style Score."""
        item_vecs = torch.stack([i['vec'] for i in items]).unsqueeze(0).to(self.device)
        item_cats = torch.tensor([i['cat'] for i in items]).unsqueeze(0).to(self.device)

        # Get the 128-dim "vibe" of this specific combination
        outfit_emb = self.stylist(item_vecs, item_cats)

        # Style Score: Inverse Euclidean distance to the User's Taste Centroid
        dist = torch.norm(outfit_emb - self.user_centroid, p=2).item()
        s_style = 1.0 / (1.0 + dist)

        # Relevance Score: Mean cosine similarity of items to the text query
        s_rel = sum(F.cosine_similarity(i['vec'].unsqueeze(0), query_vec, dim=1).item() for i in items) / len(items)

        # Final Blend
        final_score = (self.alpha * s_style) + (self.beta * s_rel)

        return final_score, outfit_emb

    def recommend(self, query_vec: torch.Tensor, context: Dict, num_recommendations: int = 5):
        """The Main Pipeline: Filter -> Combine -> Rank."""
        # 1. Prune wardrobe
        candidates = self.gatekeeper_filter(query_vec, context)

        # 2. Simple heuristic combination (e.g., Top + Bottom + Shoes)
        # Note: In production, use a more sophisticated beam search or GA for combinations
        potential_outfits = self._generate_valid_combinations(candidates)

        ranked_outfits = []
        for outfit in potential_outfits:
            score, emb = self.score_outfit(outfit, query_vec)
            ranked_outfits.append({
                'items': outfit,
                'score': score,
                'embedding': emb
            })

        ranked_outfits.sort(key=lambda x: x['score'], reverse=True)
        return ranked_outfits[:num_recommendations]

    def _generate_valid_combinations(self, candidates):
        """Helper to group items into valid T-B-S outfits."""
        tops = [c for c in candidates if c['cat'] in [1, 2, 3]]  # Simplified IDs
        bottoms = [c for c in candidates if c['cat'] in [4, 5]]
        shoes = [c for c in candidates if c['cat'] in [10]]

        outfits = []
        for t in tops[:5]:
            for b in bottoms[:5]:
                for s in shoes[:3]:
                    outfits.append([t, b, s])
        return outfits

    def update_feedback(self, outfit_emb: torch.Tensor, liked: bool, s_style: float, s_rel: float):
        """Phase 3: Real-time update of User Centroid and Bandit weights."""
        reward = 1.0 if liked else -1.0

        # Update Taste Centroid (Representational RL)
        if liked:
            # Move closer to the vibe
            self.user_centroid = (1 - self.eta) * self.user_centroid + self.eta * outfit_emb
        else:
            # Move away from the vibe
            direction = outfit_emb - self.user_centroid
            self.user_centroid = self.user_centroid - (self.eta * 0.5 * direction)

        self.user_centroid = F.normalize(self.user_centroid, p=2, dim=1)

        # Update Alpha/Beta (Behavioral RL)
        self.alpha += reward * (s_style - 0.5) * 0.05
        self.beta += reward * (s_rel - 0.5) * 0.05

        # Constraints
        total = self.alpha + self.beta
        self.alpha, self.beta = max(0.2, self.alpha / total), max(0.2, self.beta / total)


# ---------------------------------------------------------
# Execution Logic
# ---------------------------------------------------------
if __name__ == "__main__":
    # Example Initialization
    brain = FashionBrain("best_stylist.pth", "user_wardrobe.pt")

    # Simulate a user query: "I want a casual outfit for a cold day"
    # query_vec would come from your StudentCLIP embedder
    mock_query = torch.randn(1, 512)
    mock_context = {'temp_score': 0.2, 'formal_score': 0.3}

    results = brain.recommend(mock_query, mock_context)

    for i, res in enumerate(results):
        print(f"Outfit {i + 1} | Score: {res['score']:.3f} | Items: {[item['id'] for item in res['items']]}")