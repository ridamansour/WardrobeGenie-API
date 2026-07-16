# **ADR-0002: Use a Multi-Head Attribute Classifier with FashionCLIP Pseudo-Labels**


## Status

Accepted

---

## Context

WardrobeGenie enriches each detected garment with semantic attributes used throughout the recommendation pipeline. These attributes include:

* Fit
* Style
* Weather suitability
* Formality

These predictions are later consumed by the recommendation engine as hard constraints and ranking features.

The project required a scalable approach that could predict multiple attributes while minimizing computational cost during inference.

---

## Decision

WardrobeGenie uses a **single EfficientNet-B0 backbone with multiple prediction heads**, allowing all garment attributes to be predicted in a single forward pass.

Training targets are generated using **FashionCLIP-based pseudo-labeling** rather than generic CLIP or manual annotation.

---

## Rationale

### Single Model Instead of Multiple CNNs

The initial design proposed training an independent CNN for each attribute.

For example:

```text
Garment
   │
   ├── CNN → Fit
   ├── CNN → Style
   ├── CNN → Formality
   └── CNN → Weather
```

Although conceptually simple, this approach introduced several drawbacks:

* four separate training pipelines
* increased storage requirements
* duplicated feature extraction
* longer inference times
* higher deployment complexity

Instead, the project adopted a multi-task learning approach.

A shared EfficientNet-B0 backbone extracts visual features once, while independent prediction heads estimate each semantic attribute.

```text
Garment
      │
      ▼
EfficientNet-B0
      │
 ┌────┼────┬────┐
 ▼    ▼    ▼    ▼
Fit Style Weather Formality
```

This significantly reduces computational cost while allowing the backbone to learn representations shared across related fashion tasks.

---

### Initial Pseudo-Labeling with CLIP

The first dataset generation strategy used OpenAI's CLIP model.

Each garment crop was compared against handcrafted textual prompts describing each attribute.

For example:

```text
"This garment is formal."

"This garment is casual."

"This garment is suitable for winter."

"This garment is oversized."
```

Cosine similarity between image and text embeddings was then converted into soft supervision targets.

While this approach eliminated manual labeling, exploratory analysis revealed that similarity scores were frequently concentrated within a narrow range (approximately 0.4–0.6), making many predictions weakly discriminative.

The underlying reason is that the general-purpose CLIP model is optimized for broad image-text alignment across diverse domains rather than fine-grained distinctions within fashion.

For WardrobeGenie, distinguishing between garments such as different jacket styles or levels of formality proved considerably more challenging than distinguishing unrelated concepts.

---

### Manual Annotation Considered

To improve label quality, a manual labeling pipeline was designed and implemented.

This approach would have produced higher-quality supervision but required substantial human effort to annotate thousands of garment crops consistently.

Although technically feasible, manual annotation was ultimately rejected due to its scalability limitations.

---

### Migration to FashionCLIP

The project instead adopted **FashionCLIP**, a model specifically trained on fashion-related image-text pairs.

Exploratory Data Analysis (EDA) was performed on the generated pseudo-labels before integrating the model into the training pipeline.

Compared with generic CLIP, FashionCLIP produced probability distributions that were substantially more informative and better separated across semantic fashion attributes.

This resulted in stronger pseudo-labels while preserving the advantages of automated dataset generation.

---

## Alternatives Considered

#### Multiple Independent CNNs

**Pros**

* Simple architecture
* Independent optimization for each task

**Cons**

* Redundant computation
* Increased memory usage
* Multiple deployment artifacts
* Longer inference time

---

#### Generic CLIP Pseudo-Labeling

**Pros**

* Fully automated
* No manual annotation

**Cons**

* Weakly discriminative similarity scores
* Poor separation between fashion-specific attributes

---

#### Manual Annotation

**Pros**

* Highest label quality
* Complete control over supervision

**Cons**

* Time intensive
* Difficult to scale
* Requires ongoing annotation effort

---

## Consequences

### Positive

* Single forward pass predicts all semantic attributes.
* Lower computational cost than maintaining multiple attribute-specific models.
* Shared visual features improve efficiency through multi-task learning.
* FashionCLIP provides more meaningful pseudo-labels than generic CLIP while avoiding large-scale manual annotation.

### Negative

* Multi-task learning can introduce competition between tasks if one attribute dominates optimization.
* Pseudo-label quality remains dependent on the underlying teacher model.
* FashionCLIP supervision may still contain noise compared with fully curated labels.

---

## Future Considerations

Future iterations may combine FashionCLIP pseudo-labels with a smaller manually annotated validation set to further improve label quality through semi-supervised learning or active learning.