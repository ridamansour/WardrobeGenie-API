# ADR-0004: Distill FashionCLIP into a Lightweight Visual Embedding Encoder

## Status

Accepted

---

## Context

WardrobeGenie retrieves garments by comparing user queries with garment embeddings in a shared semantic vector space.

This required a visual encoder capable of producing embeddings that were semantically compatible with the text encoder while remaining efficient enough for production deployment.

Initially, OpenAI's CLIP was considered because its image and text encoders naturally share the same embedding space.

However, CLIP was trained on general-purpose internet imagery and demonstrated weaker discrimination between visually similar garments than desired.

A more fashion-specific representation was required.

---

## Decision

WardrobeGenie uses **FashionCLIP** as the teacher model for the visual embedding pipeline.

All cropped garment images are first embedded using FashionCLIP to generate 512-dimensional target vectors.

A lightweight **MobileNetV3** student network is then trained through knowledge distillation to regress these embeddings directly from garment crops.

The resulting student model replaces the significantly larger FashionCLIP model during inference while preserving the same embedding space.

---

## Evolution of the Design

### Initial Design: OpenAI CLIP

The initial implementation planned to use OpenAI CLIP directly.

Its principal advantage was that image and text embeddings already occupied the same semantic space, simplifying retrieval.

However, experimentation showed that CLIP's representations were optimized for broad semantic concepts rather than fine-grained fashion understanding.

Distinguishing between visually similar garments proved more difficult than desired.

---

### Alternative Considered: Deploy FashionCLIP Directly

Another possibility was deploying FashionCLIP itself during inference.

Although this produced high-quality embeddings, it significantly increased computational requirements and inference latency compared with lightweight CNN architectures.

For a production recommendation API, this additional cost was unnecessary.

---

### Final Design: Knowledge Distillation

Instead of deploying FashionCLIP directly, the project adopted a teacher-student approach.

First, every cropped garment image is encoded once using FashionCLIP.

The resulting embeddings are exported and stored as training targets.

A MobileNetV3 student network is then trained to reproduce those embeddings from the garment images.

During inference, only the distilled student model is required.

---

## Rationale

### Fashion-Specific Semantic Space

FashionCLIP is trained specifically for fashion imagery, producing embeddings that better capture relationships between garments than general-purpose vision-language models.

Using FashionCLIP therefore provides a stronger semantic foundation for garment retrieval.

---

### Shared Embedding Space

Because the query encoder (ADR-0005) is likewise distilled from FashionCLIP, both image and text representations occupy the same semantic embedding space.

This enables cosine similarity to compare garments and user queries directly without requiring additional projection networks.

---

### Efficient Deployment

The expensive FashionCLIP model is used only during dataset generation.

Inference relies exclusively on the lightweight MobileNetV3 student network.

This significantly reduces computational cost while preserving the semantic structure learned by the teacher model.

---

### Offline Teacher Generation

Generating teacher embeddings is naturally suited to the offline MLOps pipeline.

Each garment crop is processed once using FashionCLIP, after which the resulting embeddings become fixed supervision targets for student training.

This avoids repeatedly executing the teacher model during training or inference.

---

## Alternatives Considered

### OpenAI CLIP

**Pros**

* Shared image/text embedding space
* Mature vision-language model

**Cons**

* Not specialized for fashion
* Reduced discrimination between visually similar garments

---

### Deploy FashionCLIP Directly

**Pros**

* Highest embedding quality
* No distillation required

**Cons**

* Larger model
* Higher inference latency
* Greater computational cost

---

### CNN Classification Features

Use intermediate CNN features directly.

**Pros**

* Simple implementation

**Cons**

* Not optimized for semantic retrieval
* No shared embedding space with text

---

### FashionCLIP + MobileNetV3 Distillation (Selected)

**Pros**

* Fashion-specific semantic embeddings
* Lightweight deployment
* Shared embedding space with text
* Reduced inference cost
* Scalable production deployment

**Cons**

* Additional offline training step
* Requires generation of teacher embeddings

---

## Consequences

### Positive

* Efficient visual embedding generation.
* Better garment semantics than general-purpose CLIP.
* Compatible with semantic vector retrieval.
* Lightweight production inference.
* Offline teacher generation integrates naturally with Airflow.

### Negative

* Student quality depends on teacher quality.
* Distillation introduces an additional training pipeline.
* Updating the teacher requires regenerating embeddings.

---

## Future Considerations

Future versions may investigate newer multimodal foundation models or larger fashion-specific encoders as teachers.

Because the architecture separates teacher generation from inference, replacing the teacher requires only regenerating embeddings and retraining the student model while leaving the online serving infrastructure unchanged.

---

## Related ADRs

* **ADR-0002:** Use a Multi-Head Attribute Classifier with FashionCLIP Pseudo-Labels
* **ADR-0005:** Distill a Lightweight Semantic Query Encoder
* **ADR-0008:** Use Qdrant for Semantic Garment Retrieval
* **ADR-0009:** Learn Outfit Representations Using a Triplet-Loss Set Transformer

---

### Architecture Overview

```text
          Garment Crop
                │
                ▼
         FashionCLIP Teacher
                │
       512-dim Target Vector
                │
      (Offline Dataset Export)
                │
                ▼
       Train MobileNetV3 Student
                │
                ▼
     Lightweight Visual Encoder
                │
                ▼
        512-dim Embedding
                │
                ▼
             Qdrant
```