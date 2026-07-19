# ADR-0003: Represent Garments Using Three Dominant Colors

## Status

Accepted

---

## Context

WardrobeGenie incorporates color as part of each garment's semantic representation. Color information is later used by the recommendation engine to evaluate outfit compatibility and color harmony.

The system required a compact representation that captures the primary colors of a garment without introducing unnecessary complexity into downstream recommendation algorithms.

---

## Decision

WardrobeGenie extracts the **three dominant colors** from each garment crop using **K-Means clustering (`k=3`)**.

Each cluster centroid is converted to a hexadecimal color value and stored together with its relative proportion within the garment.

---

## Rationale

The objective is not to perfectly reproduce every color present in a garment, but rather to capture the colors that contribute most to its overall visual appearance.

A representation limited to three dominant colors provides a practical balance between expressiveness and simplicity.

In practice, most garments are visually characterized by one to three primary colors. Increasing the number of extracted colors often captures minor details, shadows, logos, stitching, or image noise that contribute little to perceived color identity.

Because the downstream color harmony algorithm compares the extracted colors between garments, introducing additional low-significance colors increases the complexity of these comparisons and can reduce the usefulness of the resulting harmony score.

Using **`k=3`** therefore provides sufficient color information while keeping the representation compact and consistent across all garments.

---

## Alternatives Considered

### Single Dominant Color (`k=1`)

**Pros**

* Extremely compact representation
* Lowest computational cost

**Cons**

* Unable to represent garments containing multiple prominent colors
* Loses important visual information

---

### Larger Number of Clusters (`k>3`)

**Pros**

* Captures finer color variation

**Cons**

* Introduces colors that are often visually insignificant
* Increases complexity of downstream harmony calculations
* More sensitive to lighting artifacts and small decorative elements

---

### Full Color Histogram

**Pros**

* Preserves the complete color distribution

**Cons**

* High-dimensional representation
* More difficult to compare efficiently
* Unnecessary for WardrobeGenie's recommendation pipeline

---

## Consequences

### Positive

* Compact and consistent color representation
* Efficient downstream color harmony calculations
* Preserves the visually dominant colors of most garments
* Minimal computational overhead

### Negative

* Fine-grained color variations may be omitted
* Lighting conditions can still influence extracted colors
* Garments with highly complex patterns may not be fully represented

---

## Future Considerations

Future iterations may evaluate perceptually uniform color spaces such as **CIELAB** or adaptive clustering strategies that dynamically adjust the number of dominant colors based on garment complexity.

---

I actually think this is a **better ADR** than "Use K-Means."

The real design decision wasn't the clustering algorithm—K-Means was simply the implementation. The architectural choice was to **compress a garment's color information into three representative colors** because that's the level of detail the downstream color harmony function actually needs. That's exactly the kind of design trade-off ADRs are meant to document.
