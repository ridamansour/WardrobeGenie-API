import unittest
import torch
# CHANGED: Using CLIPTextModelWithProjection for FashionCLIP's projection alignment
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from semantic_processing.query_vectorization.query_vectorizer import QueryVectorizer


class MyTestCase(unittest.TestCase):
    def test_something(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # CHANGED: Swapped "openai/clip-vit-base-patch32" for the FashionCLIP repository
        model_name = "patrickjohncyh/fashion-clip"
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        teacher = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
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
                # CHANGED: extract 'text_embeds' instead of pooler_output to get the 512-dim projection
                embeddings = outputs.text_embeds

            # L2 Normalization (alternative to F.normalize)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            return embeddings

        # Initialize your student model wrapper
        student = QueryVectorizer()

        # Generate vectors
        teacher_vec = encode_text(["formal black blazer"])

        # Ensure student returns a PyTorch tensor on the same device for cosine similarity
        student_vec = student.encode(["formal black blazer"])
        if not isinstance(student_vec, torch.Tensor):
            student_vec = torch.tensor(student_vec)
        student_vec = student_vec.to(device)

        # Calculate similarity
        sim = torch.cosine_similarity(teacher_vec, student_vec)
        print(f"Cosine Similarity between FashionCLIP and Student: {sim.item():.4f}")

        # Verify alignment threshold
        self.assertGreaterEqual(sim.item(), 0.90)


if __name__ == '__main__':
    unittest.main()