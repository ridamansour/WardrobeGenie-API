# ADR-0001: Use RF-DETR-Nano for Garment Detection

## Status

Accepted

---

## Context

WardrobeGenie requires reliable detection of multiple garments from a single wardrobe image before downstream processing. Detection quality directly influences attribute prediction, embedding generation, color extraction, and recommendation quality.

The detector needed to:

* detect multiple garments within a single image
* localize garments accurately for downstream cropping
* generalize to overlapping clothing items
* integrate seamlessly into the PyTorch training pipeline
* remain lightweight enough for practical deployment

Several object detectors were considered, including YOLO variants, Faster R-CNN, Grounding DINO, and **RF-DETR-Nano**.

---

## Decision

WardrobeGenie uses **RF-DETR-Nano** as its primary garment detector.

---

## Rationale

### Optimized for Feature Quality

Garment detection is only performed during wardrobe ingestion. Recommendation requests operate on precomputed garment representations rather than raw images.

This shifts the design objective from maximizing inference speed to maximizing crop quality for downstream computer vision models.

More accurate garment crops improve:

* attribute prediction
* visual embeddings
* color extraction
* recommendation quality

---

### Lightweight Transformer Architecture

RF-DETR-Nano provides many of the advantages of transformer-based detection while remaining significantly smaller than larger DETR models.

This provides a practical compromise between detection accuracy and computational cost, making it suitable for both model training and deployment on commodity GPUs.

---

### Robust Detection in Fashion Images

Fashion images frequently contain:

* overlapping garments
* layered clothing
* accessories
* multiple objects occupying nearby regions

Transformer-based detectors model relationships between objects globally, helping distinguish nearby garments without relying heavily on manually designed post-processing heuristics.

---

### Clean End-to-End Training

RF-DETR follows an end-to-end detection paradigm using bipartite matching rather than anchor assignment.

Compared with traditional detector pipelines, this reduces detector-specific heuristics and integrates naturally with the rest of WardrobeGenie's PyTorch-based training workflow.

---

### Architectural Consistency

WardrobeGenie already relies on transformer architectures for downstream recommendation through the Set Transformer.

Choosing RF-DETR-Nano maintains architectural consistency across the perception and recommendation stages while keeping the overall inference pipeline lightweight.

---

## Alternatives Considered

### YOLO (v8/v9/v10/v11)

**Pros**

* Extremely fast inference
* Mature tooling
* Excellent deployment ecosystem

**Cons**

* Relies on non-maximum suppression (NMS)
* More detector-specific tuning
* Optimized primarily for real-time detection workloads

YOLO would likely be preferred for applications requiring continuous real-time detection, such as video analytics or mobile camera inference.

---

### Faster R-CNN

**Pros**

* Well-established architecture
* Strong localization

**Cons**

* Higher inference latency
* Less competitive efficiency compared with modern lightweight detectors

---

### Grounding DINO

**Pros**

* Open-vocabulary detection
* Strong zero-shot capabilities

**Cons**

* Considerably larger model
* Higher computational requirements
* Open-vocabulary detection was outside the scope of the project

---

## Consequences

### Positive

* High-quality garment crops
* Improved downstream feature extraction
* Lightweight deployment compared with larger DETR models
* Clean integration into the PyTorch training pipeline

### Negative

* Slower than lightweight YOLO variants
* Longer training times than some CNN-based detectors
* Increased ingestion latency compared with detectors optimized purely for speed

---

## Future Considerations

If future deployments require real-time mobile inference directly on user devices, lightweight YOLO variants may be evaluated as alternative perception backends. Because the detection stage is isolated from downstream recommendation components, the detector can be replaced without modifying the remainder of the pipeline.


