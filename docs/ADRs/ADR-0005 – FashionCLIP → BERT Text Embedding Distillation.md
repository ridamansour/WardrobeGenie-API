# ADR-0004: Use Knowledge Distillation to Build a Lightweight Query Encoder

## Status
Accepted

## Context

WardrobeGenie allows users to search their wardrobe using natural language queries such as:
* "business casual for a rainy day"
* "warm outfit for winter"
* "black shoes for a formal event"

To support semantic retrieval, user queries must be converted into dense vector representations that capture both garment characteristics and contextual intent.

Because no large corpus of real user fashion searches was available, the project required a method for generating representative training data while keeping inference efficient for production deployment.

## Decision

WardrobeGenie uses a **distilled BERT encoder** trained on a synthetically generated fashion query dataset.
The dataset is automatically generated through template-based synthesis and augmented with multiple forms of linguistic variation to better approximate real-world user input.

---

## Rationale

### Knowledge Distillation

Rather than deploying a large language encoder during inference, WardrobeGenie uses **knowledge distillation** to transfer semantic understanding from a larger teacher model to a lightweight BERT student encoder.

The teacher model is **FashionCLIP's text encoder**, which provides rich semantic representations for fashion-related language. During training, the student learns to approximate the teacher's embedding space while requiring significantly fewer computational resources during inference.

This approach preserves much of the teacher model's semantic capability while reducing latency and memory requirements for the recommendation API.

---

### Synthetic Dataset Generation

Knowledge distillation requires a sufficiently diverse corpus of text examples spanning the recommendation domain. Because no large collection of real user fashion searches was available, the project generates this dataset procedurally.

Queries are synthesized through combinatorial templates that combine:

* garment categories
* colors
* occasions
* weather conditions
* seasons
* styles
* formality levels

This produces thousands of semantically distinct training examples while ensuring broad coverage of supported recommendation contexts.

Example generated queries include:

```text
black blazer for a business meeting

casual jacket for autumn

white sneakers for summer

warm hoodie for winter
```

---

### Linguistic Augmentation

Real user queries rarely follow perfectly structured templates.

To better approximate natural search behaviour, the generated dataset is augmented through several forms of linguistic noise, including:

* US and UK spelling variations

  * color → colour
* Keyboard proximity typographical errors

  * jacket → jakcet
* Random character deletions

  * winter → wintr
* Alternative vocabulary and synonyms
* Multiple sentence templates
* Variation in adjective and noun ordering

These augmentations encourage the student model to learn semantic meaning rather than memorizing template structure, improving robustness to imperfect user input.

---

### Efficient Production Inference

The distilled BERT encoder is executed for every recommendation request.

Using a lightweight student model instead of the larger FashionCLIP text encoder significantly reduces inference cost while maintaining compatibility with the embedding space used throughout the recommendation pipeline.

This provides a practical balance between semantic quality and real-time performance for API-based deployment.

---

## Future Considerations

As WardrobeGenie accumulates anonymized search queries, the synthetic dataset can be progressively supplemented with real user searches. Future versions may employ continual knowledge distillation, allowing the lightweight student encoder to periodically inherit improvements from newer teacher models while maintaining a compact deployment footprint.