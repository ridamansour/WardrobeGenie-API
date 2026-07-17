# ADR-0000: Adopt a Server-Based Inference Architecture

## Status

Accepted

**Supersedes:** Initial proposal for on-device inference

---

## Context

During the initial design phase, WardrobeGenie was envisioned as a mobile application capable of performing all machine learning inference directly on the user's device.

This approach would eliminate network latency and improve offline functionality while keeping user images local to the device.

However, the system consists of multiple machine learning models with different computational characteristics, including object detection, attribute prediction, embedding generation, and outfit ranking.

The feasibility of deploying the complete inference pipeline on mobile devices was evaluated before implementation began.

---

## Decision

WardrobeGenie adopts a **server-based inference architecture** exposed through a REST API rather than executing the machine learning pipeline directly on client devices.

Mobile and web applications communicate with the backend through HTTP endpoints, while all machine learning inference is performed on the server.

---

## Rationale

### Project Scope

Deploying the complete machine learning pipeline on-device would have significantly increased the scope of the project.

Supporting mobile inference would require additional work beyond model development, including model conversion, platform-specific optimization, hardware compatibility testing, and mobile deployment.

Given the project's timeline, these activities would have diverted effort away from the recommendation system itself.

---

### Model Compatibility

The perception and recommendation pipeline consists of several independently trained models with different architectures.

Not every model is equally suitable for mobile deployment or aggressive quantization.

Maintaining acceptable accuracy across the complete pipeline would require evaluating each model independently and potentially redesigning parts of the architecture specifically for edge devices.

---

### Development Expertise

The team's experience was primarily focused on machine learning, software architecture, and Web development.

Developing a robust on-device inference solution would require additional expertise in Android or iOS machine learning deployment, introducing further implementation risk within the available development period.

---

### Separation of Concerns

Moving inference to a centralized backend allows the client application to remain lightweight while the server manages:

* model loading
* inference
* vector search
* recommendation generation
* model updates

This architecture also enables improvements to machine learning models without requiring users to update the client application.

---

## Alternatives Considered

### Full On-Device Inference

**Pros**

* Offline functionality
* Reduced network dependency
* Improved user privacy
* Lower server compute requirements

**Cons**

* Increased implementation complexity
* Model conversion and optimization required
* Larger application size
* Higher hardware requirements
* Greater maintenance burden

---

### Hybrid Inference

A hybrid architecture was also considered where lightweight perception models would execute on the client while recommendation and vector search remained on the server.

Although technically feasible, this approach introduced additional synchronization and deployment complexity without providing sufficient benefit within the project's timeframe.

---

### Server-Based Inference (Selected)

**Pros**

* Simpler deployment
* Centralized model management
* Easier experimentation and retraining
* Consistent inference across devices
* Reduced client complexity

**Cons**

* Requires network connectivity
* Increased server resource usage
* Added inference latency compared with local execution

---

## Consequences

### Positive

* Reduced project complexity
* Faster development cycle
* Easier model updates and experimentation
* Smaller client applications
* Simplified deployment across multiple platforms

### Negative

* Continuous network connectivity is required for inference
* Backend infrastructure must scale with user demand
* Server operational costs increase as usage grows

---

## Future Considerations

The architecture intentionally isolates the client from the machine learning implementation.

If future hardware and project resources permit, selected models may be deployed on-device using frameworks such as TensorFlow Lite, Core ML, or ONNX Runtime Mobile while retaining the existing server-based API as a fallback.