import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from recomendation_engine.model import OutfitEmbeddingTransformer


class FashionBrain:
    def __init__(self, model_path: str, wardrobe_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.stylist = OutfitEmbeddingTransformer().to(self.device)
        self.stylist.load_state_dict(torch.load(model_path, map_location=self.device))
        self.stylist.eval()

        # Safely handle the absence of a local dummy pool
        self.wardrobe = []
        if wardrobe_path and os.path.exists(wardrobe_path):
            data = torch.load(wardrobe_path)
            self.wardrobe = data.get('pool', [])

        # User Centroid mapping their specific taste in the 128-dim hypersphere
        self.user_centroid = torch.randn(1, 128).to(self.device)
        self.alpha = 0.6  # Weight for Aesthetic Style
        self.beta = 0.4  # Weight for Query Relevance
        self.eta = 0.15  # Learning rate

    def gatekeeper_filter(self, query_vec: torch.Tensor, context: Dict, pool: Optional[List[Dict]] = None,
                          top_k: int = 40):
        # Allow dynamic pools (like external DB search results) to override the local wardrobe
        active_pool = pool if pool is not None else self.wardrobe

        candidates = []
        for item in active_pool:
            # Assumes your Attribute Classifier appended warmth/formality to item
            if abs(item.get('warmth', 0.5) - context.get('temp_score', 0.5)) > 0.4:
                continue
            if abs(item.get('formality', 0.5) - context.get('formal_score', 0.5)) > 0.5:
                continue

            relevance = F.cosine_similarity(item['vec'].unsqueeze(0), query_vec, dim=1).item()
            candidates.append({'item': item, 'rel': relevance})

        candidates.sort(key=lambda x: x['rel'], reverse=True)
        return [c['item'] for c in candidates[:top_k]]

    @torch.no_grad()
    def score_outfit(self, items: List[Dict], query_vec: torch.Tensor):
        item_vecs = torch.stack([i['vec'] for i in items]).unsqueeze(0).to(self.device)
        item_cats = torch.tensor([i['cat'] for i in items]).unsqueeze(0).to(self.device)

        outfit_emb = self.stylist(item_vecs, item_cats)

        dist = torch.norm(outfit_emb - self.user_centroid, p=2).item()
        s_style = 1.0 / (1.0 + dist)
        s_rel = sum(F.cosine_similarity(i['vec'].unsqueeze(0), query_vec, dim=1).item() for i in items) / len(items)

        final_score = (self.alpha * s_style) + (self.beta * s_rel)
        return final_score, outfit_emb

    def update_feedback(self, outfit_emb: torch.Tensor, liked: bool, s_style: float, s_rel: float):
        reward = 1.0 if liked else -1.0

        if liked:
            self.user_centroid = (1 - self.eta) * self.user_centroid + self.eta * outfit_emb
        else:
            direction = outfit_emb - self.user_centroid
            self.user_centroid = self.user_centroid - (self.eta * 0.5 * direction)

        self.user_centroid = F.normalize(self.user_centroid, p=2, dim=1)

        self.alpha += reward * (s_style - 0.5) * 0.05
        self.beta += reward * (s_rel - 0.5) * 0.05
        total = self.alpha + self.beta
        self.alpha, self.beta = max(0.2, self.alpha / total), max(0.2, self.beta / total)