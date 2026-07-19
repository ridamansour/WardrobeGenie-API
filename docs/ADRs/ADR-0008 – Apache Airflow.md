# ADR-0007: Separate Online Inference and Offline Machine Learning Using Apache Airflow

## Status

Accepted

---

## Context

WardrobeGenie consists of two fundamentally different workloads:

1. **Online inference**, where users expect image analysis and outfit recommendations with low latency.
2. **Offline machine learning**, including dataset generation, feature extraction, vector indexing, and model training.

Although these workloads share data and models, they have conflicting operational requirements.

The recommendation API prioritizes responsiveness and availability, whereas training pipelines prioritize throughput and may execute for minutes or hours.

Additionally, the recommendation domain is inherently subject to **data drift**. Fashion trends, garment styles, seasonal preferences, and user behavior evolve continuously, meaning the recommendation models must be periodically retrained to maintain recommendation quality.

Running these workloads within a single application would increase deployment complexity, create resource contention, and risk degrading the responsiveness of the production API.

---

## Decision

WardrobeGenie separates **online inference** from **offline machine learning**.

The production FastAPI service is responsible exclusively for serving user requests, while **Apache Airflow** orchestrates all long-running machine learning workflows through scheduled Directed Acyclic Graphs (DAGs).

This architecture ensures that machine learning models—including the **Triplet-Loss Set Transformer**—can be periodically retrained to adapt to inevitable data drift without affecting the availability or latency of the online recommendation service.

The two environments are deployed independently using separate Docker Compose stacks.

---

## Evolution of the Design

### Initial Design: Everything Inside the API

The earliest architecture executed preprocessing, dataset generation, feature extraction, and model updates alongside the FastAPI application.

Although suitable during early experimentation, this approach quickly revealed several limitations.

Long-running operations consumed resources needed by inference requests, increasing response latency and reducing API responsiveness.

As the project expanded, the API became responsible for tasks unrelated to request serving, making the system increasingly difficult to maintain.

---

### Alternative Considered: Cron Jobs

Cron jobs were evaluated for scheduling recurring workflows such as model retraining and batch ingestion.

While simple to configure, cron lacked several capabilities required for production machine learning pipelines, including:

* dependency management,
* retries,
* execution history,
* monitoring,
* failure recovery,
* and workflow visualization.

As the number of machine learning pipelines increased, maintaining shell scripts became increasingly impractical.

---

### Final Design: Apache Airflow

The project adopted Apache Airflow as the orchestration layer for all offline workflows.

Each machine learning pipeline is implemented as an independent Directed Acyclic Graph (DAG) with explicit task dependencies.

This separates orchestration from business logic while providing scheduling, monitoring, retries, logging, and reproducibility.

---

## Rationale

### Separation of Concerns

The FastAPI service is responsible only for serving user requests.

Apache Airflow is responsible only for offline machine learning workflows.

Neither component depends on the internal execution logic of the other, allowing each system to evolve independently while maintaining a clean separation of responsibilities.

---

### Isolation of Resource-Intensive Tasks

Model training, feature extraction, embedding generation, and dataset synthesis are computationally intensive workloads.

Executing these tasks independently prevents them from competing with user-facing inference requests for CPU, GPU, and memory resources.

As a result, the recommendation API remains responsive even while machine learning workflows are executing in the background.

---

### Workflow Orchestration

Machine learning pipelines naturally consist of dependent stages.

For example:

```text
Dataset
    │
    ▼
Garment Detection
    │
    ▼
Garment Cropping
    │
    ▼
Feature Extraction
    │
    ▼
Embedding Generation
    │
    ▼
Qdrant Indexing
```

Airflow models these dependencies explicitly through DAGs, ensuring each stage executes only after its prerequisites have successfully completed.

---

### Continuous Adaptation to Data Drift

Unlike traditional software systems, WardrobeGenie's recommendation models are expected to experience **continuous data drift**.

The Triplet-Loss Set Transformer learns a global Style Space from historical outfit data. As fashion trends evolve and users upload increasingly diverse wardrobes, the distribution of outfits gradually changes.

Without periodic retraining, the learned Style Space would slowly become less representative of contemporary fashion, reducing recommendation quality.

Apache Airflow provides the orchestration layer required to continuously adapt the recommendation system by automating:

* ingestion of newly available wardrobe data,
* generation of updated training datasets,
* scheduled model retraining,
* evaluation against validation datasets,
* and deployment of updated model artifacts.

This enables the recommendation engine to evolve alongside changing fashion trends while keeping the online API continuously available.

---

### Reproducibility

Every workflow execution is recorded with timestamps, execution status, logs, and task history.

This improves reproducibility and simplifies debugging compared with manually executing training scripts.

Failures can be retried from intermediate stages without restarting the entire pipeline.

---

### Modular MLOps

WardrobeGenie organizes its offline workflows into dedicated DAGs.

#### Batch Ingestion

* Imports wardrobe images.
* Performs garment detection.
* Extracts garment attributes.
* Generates visual embeddings.
* Updates the Qdrant vector database.

#### Data Synthesis

* Converts Fashionpedia into COCO datasets.
* Generates synthetic natural-language queries.
* Produces FashionCLIP pseudo-labels.
* Generates teacher embeddings for knowledge distillation.

#### Continuous Training

* Aggregates newly available outfit data and user interactions.
* Retrains the Triplet-Loss Set Transformer and supporting models.
* Validates model performance.
* Publishes updated model artifacts.
* Refreshes production models without interrupting the recommendation API.

Each workflow can evolve independently without affecting the online inference service.

---

### Independent Deployment

WardrobeGenie intentionally separates deployment into two Docker environments.

#### Online Serving Stack

* FastAPI
* Qdrant

#### Offline MLOps Stack

* Apache Airflow
* PostgreSQL
* Redis
* Worker services

This isolation allows each environment to scale according to its workload characteristics while minimizing operational coupling.

---

## Alternatives Considered

### Everything Inside FastAPI

**Pros**

* Simple deployment
* Minimal infrastructure

**Cons**

* Long-running tasks compete with inference
* Difficult workflow monitoring
* Reduced maintainability
* Poor scalability

---

### Cron Jobs

**Pros**

* Simple scheduling
* Minimal dependencies

**Cons**

* Limited dependency management
* Poor monitoring
* Difficult recovery
* Unsuitable for complex machine learning workflows

---

### Custom Python Scheduler

**Pros**

* Complete implementation control
* Lightweight

**Cons**

* Reinvents orchestration functionality
* Additional maintenance burden
* Smaller ecosystem

---

### Apache Airflow (Selected)

**Pros**

* DAG-based workflow orchestration
* Dependency management
* Automatic retries
* Execution history
* Monitoring UI
* Mature MLOps ecosystem
* Easily extensible
* Supports continual retraining to address data drift

**Cons**

* Additional infrastructure
* Higher operational complexity
* Greater learning curve than simple schedulers

---

## Consequences

### Positive

* Online inference remains isolated from offline workloads.
* Machine learning workflows become reproducible and observable.
* Individual pipelines can be rerun without affecting the production API.
* Dataset generation, feature extraction, and model training become fully automated.
* Infrastructure scales according to workload rather than application design.
* Enables continual adaptation to inevitable fashion data drift through automated retraining pipelines.

### Negative

* Additional operational overhead.
* More infrastructure to deploy and maintain.
* Increased deployment complexity compared with a single application.

---

## Future Considerations

Current retraining is schedule-driven.

Future iterations may incorporate automated data drift detection, allowing Airflow to trigger retraining only when significant changes in the underlying data distribution are observed.

Additional DAGs may also orchestrate:

* experiment tracking,
* hyperparameter optimization,
* model versioning,
* automated benchmarking,
* model registry integration,
* and rollback of underperforming models,

while preserving the separation between online inference and offline machine learning.

---

## Related ADRs

* **ADR-0000:** Adopt a Server-Based Inference Architecture
* **ADR-0001:** Use RF-DETR-Nano for Garment Detection
* **ADR-0002:** Use a Multi-Head Attribute Classifier with FashionCLIP Pseudo-Labels
* **ADR-0003:** Represent Garments Using Three Dominant Colors
* **ADR-0004:** Use Knowledge Distillation to Build a Lightweight Semantic Query Encoder
* **ADR-0005:** Learn Outfit Representations Using a Triplet-Loss Set Transformer
* **ADR-0006:** Personalize Recommendations Using a Dynamic Taste Centroid

---

## Architecture Overview

```text
                      User
                        │
                        ▼
        ┌────────────────────────────────┐
        │      Online Serving Stack      │
        │────────────────────────────────│
        │ • FastAPI                      │
        │ • Qdrant                       │
        │ • Recommendation Pipeline      │
        └────────────────────────────────┘
                    ▲              │
                    │              │
      Shared Models │              │ User Feedback
      & Embeddings  │              ▼
        ┌────────────────────────────────┐
        │      Offline MLOps Stack       │
        │────────────────────────────────│
        │ • Apache Airflow               │
        │ • Batch Ingestion DAG          │
        │ • Data Synthesis DAG           │
        │ • Continuous Training DAG      │
        └────────────────────────────────┘
                    │
                    ▼
          Updated Models & Embeddings
```la