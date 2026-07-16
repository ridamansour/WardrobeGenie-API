# ADR-0007: Separate Online Inference and Offline Machine Learning Using Apache Airflow

## Status

Accepted

---

## Context

WardrobeGenie consists of two fundamentally different workloads:

1. **Online inference**, where users expect recommendations and image analysis with low latency.
2. **Offline machine learning**, including dataset generation, feature extraction, vector indexing, and model training.

Although both workloads share data and models, they have conflicting operational requirements.

The recommendation API prioritizes responsiveness and availability, whereas training pipelines prioritize throughput and may execute for minutes or hours.

Running these workloads within the same application would increase deployment complexity, resource contention, and operational risk.

---

## Decision

WardrobeGenie separates online inference from offline machine learning.

The production API remains responsible only for serving user requests, while **Apache Airflow** orchestrates all long-running machine learning workflows through scheduled Directed Acyclic Graphs (DAGs).

The two environments are deployed independently using separate Docker Compose stacks.

---

## Evolution of the Design

### Initial Design: Everything Inside the API

The earliest architecture executed preprocessing, dataset generation, feature extraction, and model updates directly alongside the FastAPI application.

Although suitable during early experimentation, this approach quickly revealed several limitations.

Long-running operations could consume resources needed by inference requests, increasing latency and reducing responsiveness.

As the project expanded, the API also became responsible for tasks unrelated to request serving.

---

### Alternative Considered: Cron Jobs

Cron jobs were considered for scheduling recurring tasks such as model retraining and batch ingestion.

While simple to configure, cron lacked features needed for machine learning pipelines:

* task dependencies,
* retry mechanisms,
* execution history,
* monitoring,
* failure recovery,
* and workflow visualization.

Managing increasingly complex pipelines through shell scripts would become difficult to maintain.

---

### Final Design: Apache Airflow

The project adopted Apache Airflow as the orchestration layer for all offline workflows.

Each major machine learning process was implemented as an independent DAG with explicit task dependencies.

This separates orchestration from business logic while providing scheduling, monitoring, retries, and reproducibility.

---

## Rationale

### Separation of Concerns

The FastAPI service is responsible only for serving user requests.

Airflow is responsible only for offline workflows.

Neither component depends on the internal execution logic of the other.

This keeps both systems focused on their respective responsibilities.

---

### Isolation of Resource-Intensive Tasks

Model training, dataset synthesis, feature extraction, and vector generation are computationally intensive.

Executing these workloads independently prevents them from competing with user-facing inference requests for CPU, GPU, and memory resources.

The recommendation API therefore remains responsive even while new models are being trained.

---

### Workflow Orchestration

Machine learning pipelines naturally consist of dependent stages.

For example:

```text
Dataset
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

### Reproducibility

Each pipeline execution is recorded with timestamps, logs, execution status, and task history.

This improves reproducibility and simplifies debugging compared with manually executing training scripts.

Failures can be retried from intermediate stages without restarting the entire workflow.

---

### Modular MLOps

WardrobeGenie currently organizes its offline workflows into dedicated DAGs, including:

* **Batch Ingestion**

  * Imports wardrobe images
  * Performs garment detection
  * Generates embeddings
  * Updates the Qdrant index

* **Data Synthesis**

  * Converts Fashionpedia into training datasets
  * Generates synthetic natural-language queries
  * Produces teacher labels and embeddings

* **Continuous Training**

  * Aggregates user feedback
  * Retrains machine learning models
  * Validates performance
  * Publishes updated models

Each workflow can evolve independently without affecting the online recommendation service.

---

### Independent Deployment

The project intentionally separates deployment into two Docker environments.

**Online Stack**

* FastAPI
* Qdrant

**Offline Stack**

* Apache Airflow
* PostgreSQL
* Redis
* Worker services

This isolation allows each environment to be scaled and maintained independently according to its workload characteristics.

---

## Alternatives Considered

### Everything Inside FastAPI

**Pros**

* Simple architecture
* Minimal infrastructure

**Cons**

* Long-running tasks compete with inference
* Difficult to monitor workflows
* Reduced maintainability

---

### Cron Jobs

**Pros**

* Easy scheduling
* Minimal dependencies

**Cons**

* Poor dependency management
* Limited monitoring
* Difficult failure recovery
* Unsuitable for complex ML pipelines

---

### Custom Python Scheduler

**Pros**

* Complete implementation control
* Lightweight

**Cons**

* Reinvents orchestration features
* Additional maintenance burden
* Less mature ecosystem

---

### Apache Airflow (Selected)

**Pros**

* DAG-based workflow orchestration
* Dependency management
* Automatic retries
* Execution history
* Monitoring interface
* Widely adopted MLOps ecosystem
* Easily extensible

**Cons**

* Additional infrastructure
* Higher operational complexity
* Learning curve compared with simple schedulers

---

## Consequences

### Positive

* Online inference remains isolated from offline workloads.
* Machine learning workflows become reproducible and observable.
* Individual pipelines can be rerun without affecting the API.
* Data synthesis, feature extraction, and training become fully automated.
* Infrastructure scales according to workload rather than application design.

### Negative

* Additional operational overhead.
* More containers and services to maintain.
* Increased deployment complexity compared with a single application.

---

## Future Considerations

Future iterations may integrate experiment tracking, model versioning, and automated deployment directly into the Airflow pipelines.

As the project grows, additional DAGs may orchestrate evaluation benchmarks, hyperparameter optimization, drift detection, or automated rollback of underperforming models while preserving the separation between online inference and offline machine learning.

---

## Related ADRs

* **ADR-0001:** Use RF-DETR-Nano for Garment Detection
* **ADR-0002:** Use a Multi-Head Attribute Classifier with FashionCLIP Pseudo-Labels
* **ADR-0004:** Use Knowledge Distillation to Build a Lightweight Semantic Query Encoder
* **ADR-0005:** Learn Outfit Representations Using a Triplet-Loss Set Transformer
* **ADR-0006:** Personalize Recommendations Using a Dynamic Taste Centroid

---

### One thing I would add

Because your repository actually has **two Docker Compose files**, I'd include this diagram near the top of the ADR:

```text
                  User
                    │
                    ▼
      ┌─────────────────────────┐
      │  Online Serving Stack   │
      │-------------------------│
      │ FastAPI                 │
      │ Qdrant                  │
      └─────────────────────────┘
                    ▲
                    │
      Shared Models │ Shared Vectors
                    │
                    ▼
      ┌─────────────────────────┐
      │ Offline MLOps Stack     │
      │-------------------------│
      │ Apache Airflow          │
      │ Data Synthesis          │
      │ Batch Ingestion         │
      │ Continuous Training     │
      └─────────────────────────┘
```