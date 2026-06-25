"""
1.B — Multi-Attribute Classifier: PyTorch Dataset
==================================================
Wraps the .pt files produced by dataset_generation.py into a standard
torch.utils.data.Dataset for training.
"""

import os
import torch
from torch.utils.data import Dataset


class AttributeDataset(Dataset):
    """
    Loads pre-generated .pt sample files.

    Each file contains:
        image            : Tensor (3, 224, 224)  – normalized crop
        category_id      : int
        fit              : Tensor (3,)           – soft probs
        style            : Tensor (4,)           – soft probs
        weather_warmth   : float
        formality_score  : float

    Parameters
    ----------
    root : str
        Directory containing .pt files (one split: train / valid / test).
    """

    def __init__(self, root: str):
        self.root  = root
        self.files = sorted(
            [f for f in os.listdir(root) if f.endswith(".pt")]
        )
        if not self.files:
            raise RuntimeError(f"No .pt files found in {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path   = os.path.join(self.root, self.files[idx])
        sample = torch.load(path, weights_only=True)
        return {
            "image":           sample["image"],                         # (3, 224, 224)
            "category_id":     torch.tensor(sample["category_id"], dtype=torch.long),
            "fit":             sample["fit"].float(),                   # (3,)
            "style":           sample["style"].float(),                 # (4,)
            "weather_warmth":  torch.tensor(sample["weather_warmth"],  dtype=torch.float32),
            "formality_score": torch.tensor(sample["formality_score"], dtype=torch.float32),
        }
