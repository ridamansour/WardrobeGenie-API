import torch
from PIL import Image
import torchvision.transforms as T
from representation_layer.visual_embeddings.model import StudentEncoder


class GarmentEmbedder:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = StudentEncoder(embed_dim=512)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval().to(device)

        self.transform = T.Compose([
            T.Lambda(self.pad_to_square),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def pad_to_square(img):
        width, height = img.size
        new_side = max(width, height)
        result = Image.new(img.mode, (new_side, new_side), (255, 255, 255))
        result.paste(img, ((new_side - width) // 2, (new_side - height) // 2))
        return result

    def embed_crop(self, crop_image: Image.Image):
        """Processes a single PIL crop and returns the 512-dim vector."""
        tensor = self.transform(crop_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vector = self.model(tensor)
        return vector  # (1, 512)