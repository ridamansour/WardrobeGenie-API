# ADR-0008: Use Qdrant for Semantic Garment Retrieval

## Status

Accepted

---

## Context

WardrobeGenie allows users to search their wardrobe using natural language queries such as:

* "business casual for a rainy day"
* "summer wedding guest outfit"
* "black jacket for winter"

These queries cannot be satisfied using exact keyword matching alone.

Instead, garments and user queries are projected into the same **512-dimensional embedding space**, allowing semantically similar items to be retrieved even when no exact textual match exists.

The recommendation pipeline therefore required a storage layer capable of:

* storing high-dimensional embeddings,
* performing efficient nearest-neighbor search,
* supporting metadata filtering,
* scaling as wardrobes grow,
* and integrating cleanly with the recommendation pipeline.

---

## Decision

WardrobeGenie stores garment embeddings in **Qdrant**, a purpose-built vector database optimized for Approximate Nearest Neighbor (ANN) search.

Each garment is represented by:

* a 512-dimensional visual embedding,
* semantic metadata (category, formality, weather warmth, etc.),
* dominant colors,
* and image references.

When a recommendation request is received, the query is encoded into the same embedding space and submitted to Qdrant.

Candidate garments are retrieved using cosine similarity and filtered by contextual constraints before being passed to the Stylist model (ADR-0005).

---

# Evolution of the Design

## Initial Design: Relational Database

The initial concept considered storing garment vectors inside a traditional relational database.

Although metadata queries were straightforward, nearest-neighbor search over hundreds of dimensions would require scanning every stored embedding.

As wardrobe sizes increase, this approach becomes increasingly inefficient.

---

## Alternative Considered: Brute-Force Vector Search

Another option was loading all embeddings into memory and computing cosine similarity directly.

While acceptable during experimentation with small datasets, this approach scales linearly with the number of garments.

Every recommendation request would require comparing the query against every stored embedding.

This approach was rejected.

---

## Final Design: Qdrant

WardrobeGenie adopted Qdrant as the semantic retrieval engine.

Instead of exhaustively comparing every embedding, Qdrant builds an Approximate Nearest Neighbor (ANN) index using **HNSW (Hierarchical Navigable Small World)** graphs.

This allows the system to retrieve only the most relevant candidate garments before passing them to the downstream recommendation models.

---

# Rationale

## Semantic Search

Traditional databases retrieve records through exact matches.

WardrobeGenie instead retrieves garments based on semantic similarity.

For example,

```text id="f22gl9"
Query:
"smart casual jacket"

↓

Nearest Embeddings

↓

Blazer
Sports Coat
Structured Jacket
```

The retrieved garments need not share identical labels; they only need to occupy nearby regions of the learned embedding space.

---

## Efficient Candidate Retrieval

The recommendation engine evaluates combinations of garments.

Searching every garment would make the number of candidate combinations grow rapidly.

Instead, Qdrant retrieves only the **Top-K** semantically relevant garments.

The Stylist model therefore evaluates combinations drawn from a much smaller candidate pool, significantly reducing computational cost.

---

## Metadata Filtering

Semantic similarity alone is insufficient.

WardrobeGenie combines vector retrieval with metadata filtering.

For example,

```text id="r0hdb7"
weather = cold

AND

formality >= 0.6

AND

category = jacket
```

Only garments satisfying these constraints participate in semantic ranking.

This prevents recommendations that are semantically similar but contextually inappropriate.

---

## Separation of Retrieval and Ranking

Qdrant is responsible only for retrieving candidate garments.

It does **not** determine the final outfit.

The architecture deliberately separates responsibilities:

```text id="jlwmg7"
Query

↓

BERT Encoder

↓

Qdrant
(Candidate Retrieval)

↓

Gatekeeper
(Context Filtering)

↓

Triplet Set Transformer
(Compatibility Ranking)

↓

Personalization

↓

Recommendation
```

This allows retrieval and recommendation models to evolve independently.

---

## Integration with MLOps

Garment embeddings are generated offline through the Airflow pipelines (ADR-0007).

The Batch Ingestion DAG continuously:

* detects garments,
* extracts attributes,
* generates embeddings,
* and updates the Qdrant index.

As models improve or embeddings are regenerated, the vector database is refreshed without modifying the recommendation API.

---

## Scalability

Wardrobe size grows continuously as users upload new clothing.

Using an ANN index allows retrieval time to remain efficient without comparing every stored garment.

This keeps recommendation latency suitable for real-time inference even as the number of stored garments increases.

---

# Alternatives Considered

## PostgreSQL

**Pros**

* Mature ecosystem
* Excellent relational querying

**Cons**

* Not designed for high-dimensional vector search
* Poor nearest-neighbor performance at scale

---

## Brute-Force Similarity Search

**Pros**

* Simple implementation
* Exact nearest neighbors

**Cons**

* Linear complexity
* Poor scalability
* Increasing latency as wardrobes grow

---

## FAISS

**Pros**

* Excellent vector search performance
* Widely used in research

**Cons**

* Primarily a similarity search library
* Additional infrastructure required for persistence and metadata management

---

## Qdrant (Selected)

**Pros**

* Native vector database
* HNSW Approximate Nearest Neighbor indexing
* Payload metadata filtering
* REST and gRPC APIs
* Persistent storage
* Docker deployment
* Well suited to semantic search workloads

**Cons**

* Additional infrastructure component
* Approximate rather than exact nearest-neighbor search

---

# Consequences

## Positive

* Efficient semantic garment retrieval.
* Candidate search scales well as wardrobes grow.
* Retrieval is separated from recommendation.
* Supports hybrid search through vectors and metadata.
* Integrates naturally with Airflow ingestion pipelines.

## Negative

* Additional service to deploy and maintain.
* Approximate search may not always return the mathematically exact nearest neighbors.
* Embeddings must remain synchronized with the vector index.

---

# Future Considerations

Future iterations may explore:

* hybrid sparse+dense retrieval,
* personalized vector indexes,
* multimodal search combining image and text queries,
* automatic embedding versioning,
* or distributed Qdrant clusters for larger deployments.

The current architecture intentionally favors efficient semantic retrieval while remaining compatible with future retrieval improvements.

---

## Related ADRs

* **ADR-0004:** Use Knowledge Distillation to Build a Lightweight Semantic Query Encoder
* **ADR-0005:** Learn Outfit Representations Using a Triplet-Loss Set Transformer
* **ADR-0006:** Personalize Recommendations Using a Dynamic Taste Centroid
* **ADR-0007:** Separate Online Inference and Offline Machine Learning Using Apache Airflow

---

## Architecture Overview

```text
              User Query
                   │
                   ▼
            BERT Student Encoder
                   │
           512-dim Query Vector
                   │
                   ▼
      ┌──────────────────────────┐
      │         Qdrant           │
      │──────────────────────────│
      │ HNSW ANN Search          │
      │ Metadata Filtering       │
      └──────────────────────────┘
                   │
           Top-K Candidate Items
                   │
                   ▼
          Gatekeeper Filtering
                   │
                   ▼
     Triplet Set Transformer
                   │
                   ▼
          Personalized Ranking
                   │
                   ▼
        Recommended Outfit
```