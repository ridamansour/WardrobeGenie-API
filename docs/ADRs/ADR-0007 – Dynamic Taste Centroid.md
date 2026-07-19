# ADR-0006: Personalize Recommendations Using a Dynamic Taste Centroid

## Status

Accepted

---

## Context

WardrobeGenie aims to continuously adapt outfit recommendations to each individual user.

While the Stylist model (ADR-0005) learns a **global Style Space** describing outfit compatibility, different users naturally prefer different regions of that space. The recommendation system therefore required a personalization strategy capable of:

* learning continuously from user feedback,
* operating without retraining the recommendation model,
* remaining computationally lightweight,
* supporting cold-start users,
* separating personalization from representation learning,
* and scaling to large numbers of users.

Several approaches were explored before arriving at the final design.

---

## Decision

WardrobeGenie represents each user by a **128-dimensional Taste Centroid** within the Style Space learned by the Triplet-Loss Set Transformer.

User interactions update this centroid using an **Exponential Moving Average (EMA)**, while a lightweight contextual-bandit mechanism dynamically adjusts the balance between **personal style compatibility** and **query relevance** during recommendation ranking.

The recommendation model itself remains fixed; only the user's representation evolves over time.

---

# Evolution of the Design

## Initial Design: Static Recommendation

The earliest recommendation pipeline ranked outfits using only a weighted combination of compatibility and query relevance.

```text
final_score =
α × compatibility +
β × relevance
```

The weighting coefficients were globally fixed.

Although this produced reasonable recommendations, every user received nearly identical rankings given the same wardrobe and query.

The system had no concept of individual taste.

---

## Second Design: Adaptive Weighting

The next iteration introduced online feedback by updating the weighting coefficients.

Users gradually influenced:

* α (importance of style)
* β (importance of query relevance)

through likes and dislikes.

While this changed recommendation behaviour, it still failed to answer the more fundamental question:

> **What styles does this particular user actually prefer?**

The recommendation engine still lacked any persistent representation of user taste.

---

## Alternative Considered: Per-User Model Fine-Tuning

Another approach considered was continually fine-tuning the recommendation model for each user.

Although theoretically capable of learning highly personalized recommendations, this approach introduced significant practical limitations:

* each user would require an independent model,
* continual retraining would become computationally expensive,
* catastrophic forgetting could occur,
* deployment complexity would increase substantially,
* and scaling to many users would become impractical.

This approach was rejected.

---

## Final Design: Taste Centroid

Rather than modifying the recommendation model itself, WardrobeGenie represents each user as a point within the learned Style Space.

Every generated outfit already has a 128-dimensional embedding.

The user's centroid represents the average region of the embedding space corresponding to their evolving preferences.

Positive interactions pull the centroid toward an outfit.

Negative interactions move it away.

The recommendation model remains unchanged while the user's representation continuously evolves.

---

## Dependency on the Stylist Model

The Taste Centroid design was made possible by the architectural decision to represent outfits as points within a learned **128-dimensional Style Space** (ADR-0005).

Because every outfit is encoded into a continuous embedding, users can likewise be represented as vectors within the same space.

Personalization therefore becomes a geometric problem—learning where a user's preferences lie—rather than a model optimization problem requiring continual retraining.

Had the Stylist instead produced only scalar compatibility scores, this representation would not have been possible. Scalar scores provide no notion of distance or direction between outfits, preventing meaningful updates based on user interactions.

The learned Style Space therefore serves as the shared foundation that enables efficient online personalization.

---

## Rationale

### Separation of Concerns

The recommendation architecture deliberately separates **representation learning** from **personalization**.

The Stylist model learns **what styles exist** by embedding outfits into a shared Style Space.

The personalization layer learns **which region of that Style Space a user prefers** by maintaining and updating a Taste Centroid.

This separation allows the recommendation model to remain user-independent while personalization evolves continuously through lightweight vector updates rather than expensive model retraining.

It also allows improvements to the Stylist model and the personalization layer to be developed independently, provided they continue to share the same embedding space.

---

### Efficient Online Learning

Updating a user's preferences requires only simple vector arithmetic.

No gradients are computed.

No neural network retraining occurs.

Each interaction immediately influences future recommendations with negligible computational cost, making personalization suitable for real-time deployment.

---

### Stable Preference Updates

WardrobeGenie updates the user's representation using an Exponential Moving Average.

For positive feedback:

[
V_{user} = (1-\gamma)V_{user} + \gamma V_{outfit}
]

For negative feedback, the centroid is shifted away from the disliked outfit using a smaller update step to avoid unstable oscillations.

After every update, the centroid is L2-normalized to remain within the learned Style Space.

EMA provides smooth adaptation, allowing long-term preferences to evolve gradually without overreacting to isolated interactions.

---

### Learning How Users Shop

Personalization extends beyond identifying preferred styles.

Different users also exhibit different search behaviour.

Some prioritize exact semantic matching to their search query.

Others are willing to sacrifice relevance for aesthetically stronger outfit recommendations.

To accommodate this, WardrobeGenie maintains two adaptive ranking weights:

* **α** — importance of personal style compatibility
* **β** — importance of query relevance

A lightweight contextual-bandit updates these weights using observed user feedback, allowing recommendation behaviour to evolve alongside user preferences.

---

### Cold-Start Support

New users begin from a neutral centroid initialized from a generic fashion representation.

As interactions accumulate, the centroid gradually shifts toward the user's preferred region of the Style Space.

This allows personalization to emerge naturally without requiring onboarding questionnaires or manual preference selection.

---

### Scalability

Each user is represented only by a compact 128-dimensional vector and two ranking coefficients.

The memory footprint remains constant regardless of recommendation model complexity.

Scaling personalization therefore requires storing user state rather than maintaining separate neural networks, making the architecture practical for large numbers of users.

---

## Alternatives Considered

### Static Global Ranking

**Pros**

* Extremely simple implementation
* No user state required

**Cons**

* No personalization
* Identical recommendations for similar wardrobes

---

### Adaptive Weighting Only

Update only α and β based on feedback.

**Pros**

* Lightweight
* Easy implementation

**Cons**

* Learns recommendation behaviour rather than user taste
* Cannot represent stylistic preferences

---

### Per-User Model Fine-Tuning

**Pros**

* Highly personalized recommendations

**Cons**

* High computational cost
* Continual retraining
* Difficult deployment
* Poor scalability

---

### Matrix Factorization

**Pros**

* Widely used recommendation technique
* Effective with large interaction datasets

**Cons**

* Requires substantial historical interaction data
* Weak cold-start performance
* Does not leverage semantic outfit embeddings

---

### Full Reinforcement Learning Agent

Model recommendation as a sequential reinforcement learning problem.

**Pros**

* Highly adaptive
* Can optimize long-term reward

**Cons**

* Significantly greater implementation complexity
* Requires substantially more interaction data
* Difficult reward engineering

The additional complexity was not justified by the project's objectives.

---

## Consequences

### Positive

* Personalization occurs immediately after each interaction.
* No retraining of the recommendation model is required.
* The recommendation model remains shared across all users.
* User preferences evolve smoothly over time.
* Cold-start users are naturally supported.
* Recommendation behaviour adapts both stylistically and semantically.
* Personalization scales efficiently through compact user representations.

### Negative

* A single centroid assumes preferences occupy one dominant region of the Style Space.
* Learning behaviour depends on the choice of EMA learning rate.
* Users with multiple unrelated fashion preferences may not be perfectly represented by a single centroid.

---

## Future Considerations

Future iterations may replace the single Taste Centroid with richer preference models, including:

* multiple preference centroids,
* Gaussian preference distributions,
* memory-based preference representations,
* sequential user encoders,
* or graph-based preference models capable of representing multiple evolving fashion identities.

The current architecture intentionally favors simplicity, interpretability, and computational efficiency while remaining fully compatible with these future extensions.

---

## Related ADRs

* **ADR-0001:** Use RF-DETR-Nano for Garment Detection
* **ADR-0002:** Use a Multi-Head Attribute Classifier with FashionCLIP Pseudo-Labels
* **ADR-0003:** Represent Garments Using Three Dominant Colors
* **ADR-0004:** Use Knowledge Distillation to Build a Lightweight Semantic Query Encoder
* **ADR-0005:** Learn Outfit Representations Using a Triplet-Loss Set Transformer

---

### Architecture Overview

```text
                   Outfit
                      │
                      ▼
           Triplet Set Transformer
                      │
          128-dim Style Embedding
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   Taste Centroid          Recommendation
     (User State)              Ranking
          │                       │
          └───────────┬───────────┘
                      ▼
               Final Recommendation
```