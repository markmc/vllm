# RFC: Clarifying vLLM Shutdown Semantics

## Introduction

This RFC defines the expected behavior of **vLLM during shutdown**, addressing two distinct use cases:
1. **Library API usage** (primarily offline inference).
2. **Online serving usage** (long-running HTTP server deployments).

Documenting these expectations will reduce surprises for users and help us achieve a consistent implementation across all features.

---

## Library API Shutdown

Library users expect deterministic, resource-safe cleanup when disposing of LLM instances.

### Expectations
* Library users may create and destroy `vllm.LLM` instances.
* Deleting an instance should:
  - Release all associated resources (memory, file descriptors, temporary files, etc.).
  - Shut down any child processes gracefully.
* No library should define global signal handlers.
  - Signal handlers are global and may interfere with applications embedding vLLM.
  - Applications must configure their own signal-handling behavior.

---

## Online Serving Shutdown

Online serving users expect graceful termination semantics, consistent with production environments like Kubernetes, where shutdown is routine due to auto-scaling or rolling upgrades.

Note - vLLM does not support any partial restart scenario, for example a "reload model" or "restart workers" capability.

### Context
* `vllm serve` launches an HTTP server for long-running online inference.
* Each request is relatively expensive (100ms+).
* Deployments typically involve multiple replicas behind a load balancer (e.g. Kubernetes Service).

### Expectations
* **Signal Handling**
  - Parent process receives a `SIGTERM` on shutdown.
  - `SIGINT` (e.g. Ctrl-C) is treated the same as `SIGTERM`.
  - Signals are sent only to the parent; child processes must not act on signals directly.

* **Graceful Quiescence**
  - Stop accepting new HTTP requests immediately.
  - Allow in-flight requests to complete, bounded by a configurable max wait time.
  - Afterward, instruct child processes to exit.

* **Child Processes**
  - Child processes should not attempt to complete requests themselves; only the parent tracks request lifecycle.
  - Child processes may delay shutdown for other reasons (e.g. KV transfer).
  - Each process is responsible for reaping its own children via `waitpid()`.

* **Resource Cleanup**
  - Close network sockets, release locks, delete temporary files, etc.

---

## Kubernetes Integration

Kubernetes shutdown semantics should align naturally with vLLM’s shutdown design.

* When a Pod enters **Terminating**:
  - Kubernetes removes its endpoints from the Service LB → no new traffic.
  - The container runtime sends `SIGTERM` to the container’s parent process.
  - After `terminationGracePeriodSeconds` (default 30s), `SIGKILL` is sent if still running.
  - Optionally, a `preStop` hook can be used to delay `SIGTERM` (e.g. sleep 5s) to allow straggling requests before quiescence.

---

## KV Transfer Considerations

In disaggregated deployments (e.g. **NIXL prefill/decode**):

* Prefill workers may need to delay shutdown until **KV transfer** is complete.
* The `KVConnector.shutdown()` method should allow the connector to defer worker process (and parents) shutdown until all pending KV transfers complete or time out.

---

## Open Questions

* **Online serving API**: consider how shutdown guarantees are reflected in the library API for online serving.
* **Ray deployments**: do Ray users have specific shutdown expectations?
* **Observability**: should the "I'm shutting down" status of the API server be observable, e.g. via the /health API?
