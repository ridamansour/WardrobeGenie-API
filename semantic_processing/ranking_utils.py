import torch
import torch.nn.functional as F


# --------------------------------------------------
# Extract scores from garment dict
# --------------------------------------------------

def garment_context_score(garment, filters, context=None):
    attr = garment["attributes"]

    # Formality
    if attr["formality_score"] < filters["min_formality"]:
        return 0.0

    # Warmth
    if attr["weather_warmth"] > filters["max_warmth"]:
        return 0.0

    # Sportiness (athletic style prob)
    sporty_score = attr["style"][2]  # athletic probability

    if not filters["sporty_ok"] and sporty_score > 0.6:
        return 0.0

    # Temperature context
    if context and "temperature" in context:
        temp = context["temperature"]

        if temp > 28 and attr["weather_warmth"] > 0.6:
            return 0.0

        if temp < 10 and attr["weather_warmth"] < 0.3:
            return 0.0

    return 1.0


# --------------------------------------------------
# Semantic similarity
# --------------------------------------------------

def garment_similarity_score(garment, query_vec):
    """
    Converts garment attributes into a lightweight embedding
    and compares to query embedding.
    """

    attr = garment["attributes"]

    # lightweight semantic vector
    garment_vec = torch.tensor([
        attr["weather_warmth"],
        attr["formality_score"],
        attr["style"][1],  # casual prob
        attr["style"][2],  # athletic prob
        attr["style"][0],  # formal prob
    ], dtype=torch.float32)

    garment_vec = F.normalize(garment_vec, dim=0)

    # project query vector down to same dimension
    query_vec_small = query_vec[:, :5]
    query_vec_small = F.normalize(query_vec_small, dim=1)

    sim = F.cosine_similarity(
        garment_vec.unsqueeze(0),
        query_vec_small
    ).item()

    return (sim + 1) / 2  # normalize to 0–1


# --------------------------------------------------
# Final garment score
# --------------------------------------------------

def garment_score(
    garment,
    query_vec,
    filters,
    context=None,
    alpha=0.6,
    beta=0.4
):
    context_score = garment_context_score(garment, filters, context)

    if context_score == 0:
        return 0.0

    sim_score = garment_similarity_score(garment, query_vec)

    return alpha * context_score + beta * sim_score