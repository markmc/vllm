# EPLB Observability Gaps - Minimal MVP

**Date**: 2025-12-09
**Context**: vLLM deployed in replicated distributed system behind inference scheduler/router/load-balancer
**Goal**: Export metrics to enable intelligent scheduling decisions at the load balancer layer

---

## Current State Summary

### What EPLB Does
- Dynamically replicates heavily-used experts across EP ranks to balance load
- Tracks expert load via token counts per physical expert
- Rebalances experts at configurable intervals (default: every 3000 steps)
- Maintains internal state: `expert_load_pass`, `expert_load_window`, `physical_to_logical_map`, etc.

### What Exists Today
**Internal tracking**: Rich tensors tracking load, mappings, balancedness
**Logging**: Optional stdout logging when `--eplb-log-balancedness` flag set:
```
EPLB step: 2500: avg_tokens=1024.50, max_tokens=2048, balancedness=0.5000
```

**Problem**: NO Prometheus metrics exposed. Logs are ephemeral and not queryable by load balancer.

### Metrics Infrastructure
- v1 engine has robust metrics pipeline: `SchedulerStats` → `PrometheusStatLogger`
- Clean patterns exist: `SpecDecodingStats`, `KVConnectorStats`, `PrefixCacheStats`
- EPLB just needs to plug into this existing pipeline

---

## Most Glaring Gaps for Load Balancer (Priority Order)

### 1. **No Instance-Level Capacity Signal** ⚠️ CRITICAL

**Problem**: Load balancer has no visibility into whether a vLLM instance can accept more work or is at capacity.

**Missing Metrics**:
```python
# Tokens processed across EP ranks this interval (raw measurements)
vllm:eplb_avg_tokens_per_rank{model}  # Gauge
vllm:eplb_max_tokens_per_rank{model}  # Gauge
```

**Why Critical**:
- Load balancer needs to know if instance is healthy or struggling
- High imbalance means one EP rank is significantly overloaded compared to others
- Absolute load levels matter too: same imbalance ratio at different scales = different capacity
- Instance with high imbalance should receive LESS traffic, not more

**Why Raw Values Instead of Pre-Computed Ratio**:
- **Absolute scale matters**: `avg=100, max=200` vs `avg=10000, max=20000` both have same ratio (0.5), but very different capacity/headroom
- **Flexible alerting**: Can compute relative imbalance (`max/avg`), absolute gap (`max - avg`), or hybrid strategies
- **Correlation**: Can cross-reference with throughput, queue depth, or other metrics
- **Prometheus best practice**: Export measurements, not calculations (preserves maximum information)

**Use Cases**:
```promql
# Relative imbalance: avoid instances where one rank is 2x overloaded
(vllm:eplb_max_tokens_per_rank / vllm:eplb_avg_tokens_per_rank) > 2.0

# Absolute imbalance: avoid instances with severe token gap
(vllm:eplb_max_tokens_per_rank - vllm:eplb_avg_tokens_per_rank) > 5000

# Hybrid: high load AND poor balance (most dangerous)
vllm:eplb_max_tokens_per_rank > 15000
AND (vllm:eplb_max_tokens_per_rank / vllm:eplb_avg_tokens_per_rank) > 1.5

# Compute traditional imbalance ratio if needed
1 - (vllm:eplb_avg_tokens_per_rank / vllm:eplb_max_tokens_per_rank)
```

---

### 2. **No Rebalancing Activity Visibility** ⚠️ HIGH

**Problem**: Load balancer doesn't know when EPLB is actively rebalancing (expensive operation).

**Missing Metrics**:
```python
# Rebalancing events (spikes indicate thrashing or load shifts)
vllm:eplb_rebalance_events_total{model}  # Counter

# Time spent rebalancing (impacts latency)
vllm:eplb_rebalance_duration_seconds{model}  # Histogram

# Rebalancing in progress (binary flag)
vllm:eplb_rebalancing{model}  # Gauge: 0 or 1
```

**Why High Priority**:
- Rebalancing is expensive (tensor reshuffling, potentially stalls inference)
- Load balancer should avoid routing heavy traffic during rebalance
- Frequent rebalancing (thrashing) indicates instance instability

**Use Case**:
```promql
# Alert on thrashing
rate(vllm:eplb_rebalance_events_total[5m]) > 2

# Reduce traffic to instance currently rebalancing
vllm:eplb_rebalancing == 1
```

---

### 3. **Existing Metrics Are Sufficient** ✅

**Request-level metrics already exposed** (for load balancer):
```python
vllm:num_requests_running{model}       # Current active requests
vllm:num_requests_waiting{model}       # Queued requests
vllm:kv_cache_usage_perc{model}        # KV cache pressure (0-100%)
vllm:e2e_request_latency_seconds       # P50/P95/P99 latency
```

**These are good!** Load balancer can already use:
- `num_requests_waiting` - route to instance with shortest queue
- `kv_cache_usage_perc` - avoid instances with high memory pressure
- `e2e_request_latency_seconds` - avoid slow instances (P95/P99)

**Gap**: No integration with EPLB state. Recommendation:
- Cross-reference EPLB load distribution with `num_requests_waiting`
- High imbalance + high queue = instance is struggling with EP load distribution

**Example combined query**:
```promql
# Instances with high queue AND severe EP rank imbalance
vllm:num_requests_waiting > 10
AND (vllm:eplb_max_tokens_per_rank / vllm:eplb_avg_tokens_per_rank) > 2.0
```

---

## High-Cardinality Metrics Considered but Deferred

**Per-expert metrics** were evaluated but **not recommended** for initial implementation due to extreme cardinality:

- `vllm:expert_replica_count{model, layer, expert_id}` - Replication factor per expert
- `vllm:expert_tokens_total{model, layer, expert_id}` - Token counts per expert

**Cardinality impact**: DeepSeek-R1 has 256 experts × 60 layers = **15,360 time series per instance**. With 100 replicas, this would create **1.5M time series** just for expert-level metrics.

**Decision**: Defer until external project demonstrates concrete use case showing significant performance improvement from workload-aware routing based on expert-level metrics. Instance-level metrics (avg/max tokens per rank) provide sufficient signal for load balancing without cardinality explosion.

**Aggregated alternatives** (if needed later):
- `vllm:eplb_total_replicas{model}` - Total replica count (single gauge, no per-expert breakdown)
- `vllm:expert_tokens_by_layer_total{model, layer}` - Aggregate to layer level only (60 time series vs 15K)

---

## Implementation Plan

### Phase 1: Instance Health Signals - **MUST HAVE**

**Add to `vllm/v1/metrics/stats.py`**:
```python
@dataclass
class EPLBStats:
    """EPLB load balancing statistics."""
    balancedness: float = 0.0              # 0.0-1.0 (1.0 = perfect balance)
    avg_tokens_per_rank: float = 0.0
    max_tokens_per_rank: int = 0
    num_rebalance_events: int = 0          # This interval
    is_rebalancing: bool = False
```

**Add to `SchedulerStats`**:
```python
eplb_stats: EPLBStats | None = None
```

**Expose to Prometheus** (in `PrometheusStatLogger`):
```python
# Raw measurements (all metrics for Phase 1)
vllm:eplb_avg_tokens_per_rank        # Gauge
vllm:eplb_max_tokens_per_rank        # Gauge
vllm:eplb_rebalance_events_total     # Counter
vllm:eplb_rebalancing                # Gauge: 0 or 1
```

**Load Balancer Usage**:
```python
# Routing weight = baseline_weight * penalties
# Compute imbalance ratio from raw metrics
imbalance_ratio = 1.0 - (avg_tokens / max_tokens)

# Penalize based on relative imbalance
if imbalance_ratio > 0.4:
    weight *= 0.5

# Penalize based on absolute load
if max_tokens_per_rank > 20000:
    weight *= 0.6

# Heavily penalize actively rebalancing instances
if eplb_rebalancing == 1:
    weight *= 0.3
```

**Example PromQL for routing decisions**:
```promql
# Score each instance (lower is better, penalize imbalanced/overloaded)
(vllm:eplb_max_tokens_per_rank / 1000)
* (vllm:eplb_max_tokens_per_rank / vllm:eplb_avg_tokens_per_rank)
+ (vllm:eplb_rebalancing * 10000)
```

**Implementation complete after Phase 1** - provides all critical signals for intelligent load balancing.

---

## Recommended First Step (Absolute Minimum)

**Two metrics to start**: `vllm:eplb_avg_tokens_per_rank` + `vllm:eplb_max_tokens_per_rank`

These two gauges give the load balancer maximum flexibility:

**Relative imbalance** (computed in PromQL):
```promql
# Imbalance ratio (0.0-1.0, higher = more imbalanced)
1 - (vllm:eplb_avg_tokens_per_rank / vllm:eplb_max_tokens_per_rank)

# Interpretation:
# < 0.2  → healthy and balanced
# 0.2-0.4 → moderate imbalance
# > 0.4  → severe imbalance, avoid routing
# > 0.6  → critical imbalance, may be unhealthy
```

**Absolute load** (direct from metrics):
```promql
# High absolute load regardless of balance
vllm:eplb_max_tokens_per_rank > 20000

# Low load with some imbalance (less concerning)
vllm:eplb_max_tokens_per_rank < 5000
```

**Why Both Matter**:
- Instance A: `avg=100, max=200` (50% imbalanced, but lightly loaded - safe to route)
- Instance B: `avg=10000, max=20000` (50% imbalanced, heavily loaded - avoid!)

**Implementation**:
1. Modify `EplbState.step()` to return `avg_tokens` and `max_tokens` values
2. Pass to scheduler → `SchedulerStats`
3. Add two gauges in `PrometheusStatLogger`:
   ```python
   gauge_eplb_avg_tokens = self._gauge_cls(
       name="vllm:eplb_avg_tokens_per_rank",
       documentation="Average tokens per EP rank (current interval)",
       multiprocess_mode="mostrecent",
       labelnames=labelnames,
   )

   gauge_eplb_max_tokens = self._gauge_cls(
       name="vllm:eplb_max_tokens_per_rank",
       documentation="Maximum tokens across EP ranks (current interval)",
       multiprocess_mode="mostrecent",
       labelnames=labelnames,
   )
   ```
4. Set values directly from `eplb_stats.avg_tokens_per_rank` and `eplb_stats.max_tokens_per_rank`

**Effort**: ~4 hours for experienced vLLM developer

---

## Non-Goals (Out of Scope)

**Architectural mismatches** (from original requirements doc - assume MoE microservices architecture):
- ❌ Router queue metrics (vLLM doesn't have separate router component)
- ❌ Per-replica queue length (EPLB doesn't maintain per-replica queues)
- ❌ Replica selection counters (routing is internal to worker, not exposed)

**High-cardinality metrics** (deferred pending external demand):
- ❌ Per-expert replica counts (`expert_replica_count{expert_id}`)
- ❌ Per-expert token counters (`expert_tokens_total{expert_id}`)
- ❌ Per-expert processing duration histograms

**Visualization/operations** (do later):
- ❌ Grafana dashboards (metrics first, visualization later)
- ❌ Alert rules (wait until metrics are deployed and baselined)

**Rationale**: vLLM's monolithic architecture means actionable signals are at the **instance level** (avg/max tokens per rank), not per-component or per-expert. High-cardinality metrics create 1.5M+ time series with unclear ROI.

---

## Success Criteria

Load balancer can answer:
1. ✅ **Is this vLLM instance healthy?** → Compute `1 - (avg_tokens / max_tokens)` to get imbalance ratio
2. ✅ **Does this instance have capacity?** → Check absolute load via `max_tokens_per_rank` and request queue
3. ✅ **Is this instance currently busy rebalancing?** → Check `eplb_rebalancing`
4. ✅ **How much headroom does this instance have?** → Check `num_requests_waiting` + `kv_cache_usage_perc` + `max_tokens_per_rank`
5. ✅ **Is this instance stable or thrashing?** → Check `rate(eplb_rebalance_events_total)`

---

## Files to Modify

1. **`vllm/v1/metrics/stats.py`**
   - Add `EPLBStats` dataclass
   - Add `eplb_stats: EPLBStats | None` field to `SchedulerStats`

2. **`vllm/distributed/eplb/eplb_state.py`**
   - Modify `step()` method to return stats (avg_tokens, max_tokens, rebalance events)

3. **`vllm/v1/core/sched/scheduler.py`**
   - Populate `eplb_stats` in `make_stats()` method

4. **`vllm/v1/metrics/loggers.py`**
   - Add EPLB metrics to `PrometheusStatLogger.__init__()`
   - Add logging to `LoggingStatLogger`

**Estimated effort**: ~150 lines total, ~4 hours for experienced vLLM developer

---

## Open Questions

1. **Rebalancing frequency**: How often does EPLB rebalance in production workloads?
   - Default is every 3000 steps
   - Impacts how often `eplb_rebalancing` flag will be 1

2. **Metrics export format**: Does the load balancer scrape Prometheus, or is push-based export needed (e.g., OTLP)?

3. **Threshold tuning**: What values constitute "severe imbalance" for routing decisions?
   - Suggestion: `max/avg > 2.0` (one rank processing 2x average load)
   - Needs validation against production traffic patterns

---

## Appendix: Why Original Requirements Don't Apply

The original `EPLB_OBSERVABILITY_REQUIREMENTS.txt` assumes a microservices architecture with:
- Separate **router** service (request entry point)
- Separate **scheduler** service (placement decisions)
- Separate **expert service** (per-expert workers)

**vLLM's architecture**:
- **Monolithic engine**: Scheduler + worker integrated
- **EP parallelism**: Experts sharded across GPUs in same process
- **EPLB**: Internal to worker, not externally routable

So metrics like `router_queue_size`, `scheduler_expert_replica_selection`, `expert_queue_length` don't map cleanly to vLLM's design. The actionable signals for load balancing are at the **instance level**, not per-component.
