import unittest
import torch

from transformers import CLIPTextModel, CLIPTokenizer, DistilBertModel, DistilBertTokenizer
from semantic_processing.query_vectorization.query_vectorizer import QueryVectorizer


class MyTestCase(unittest.TestCase):
    def test_something(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        teacher = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        teacher.eval()

        def encode_text(texts):
            tokens = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = teacher(**tokens)
                embeddings = outputs.pooler_output

            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            return embeddings

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        teacher = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        student = QueryVectorizer()

        teacher_vec = encode_text(["formal black blazer"])
        student_vec = student.encode(["formal black blazer"])

        sim = torch.cosine_similarity(teacher_vec, student_vec)
        print(sim.item())
        self.assertGreaterEqual(sim.item(), 0.90)


if __name__ == '__main__':
    unittest.main()
