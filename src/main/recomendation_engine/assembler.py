import itertools
import random

import torch


def assemble_plausible_outfits(candidates, max_outfits=100):
    """
    Groups candidates by category and creates valid Top + Bottom + Shoe sets.
    """
    by_cat = {}
    for item in candidates:
        cat = item['category']
        if cat not in by_cat: by_cat[cat] = []
        by_cat[cat].append(item)

    # Basic logic: We need at least a top and a bottom
    tops = by_cat.get('top', [])
    bottoms = by_cat.get('bottom', [])
    shoes = by_cat.get('shoes', [])

    outfits = []
    # Cartesian product of available categories
    # Limit to max_outfits to keep inference fast
    for t, b, s in itertools.product(tops, bottoms, shoes):
        outfits.append({
            "items": [t, b, s],
            "vectors": torch.stack([t['vector'], b['vector'], s['vector']]),
            "categories": torch.tensor([t['cat_id'], b['cat_id'], s['cat_id']])
        })
        if len(outfits) >= max_outfits: break

    return outfits