import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class FashionpediaCropDataset(Dataset):
    def __init__(self, split_root):
        self.split_root = split_root
        ann_path = os.path.join(split_root, "_annotations.coco.json")
        self.emb_dir = os.path.join(split_root, "teacher_embeddings")

        with open(ann_path, 'r') as f:
            data = json.load(f)

        self.images = {img['id']: img['file_name'] for img in data['images']}
        self.annotations = data['annotations']

        self.student_transform = T.Compose([
            T.Lambda(self.pad_to_square),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def pad_to_square(img):
        width, height = img.size
        new_side = max(width, height)
        result = Image.new(img.mode, (new_side, new_side), color=(255, 255, 255))
        result.paste(img, ((new_side - width) // 2, (new_side - height) // 2))
        return result

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        ann_id = ann['id']
        img_filename = self.images.get(ann['image_id'])

        img_path = os.path.join(self.split_root, img_filename)
        emb_path = os.path.join(self.emb_dir, f"{ann_id}.pt")

        if not os.path.exists(img_path) or not os.path.exists(emb_path):
            return None

        try:
            # Load Student Image
            full_img = Image.open(img_path).convert("RGB")
            x, y, w, h = ann['bbox']
            cropped_img = full_img.crop((x, y, x + w, y + h))
            student_tensor = self.student_transform(cropped_img)

            # Load Pre-computed Teacher Embedding
            teacher_tensor = torch.load(emb_path)

            return student_tensor, teacher_tensor
        except Exception:
            return None