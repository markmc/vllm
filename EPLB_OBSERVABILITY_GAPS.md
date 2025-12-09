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

### 3. **No Per-Expert Replication Visibility** 🔶 MEDIUM

**Problem**: Load balancer can't understand instance capacity by expert usage patterns.

**Missing Metrics**:
```python
# Current replication factor per expert
vllm:expert_replica_count{model, layer, expert_id}  # Gauge

# Aggregated view: total replicas across all experts
vllm:eplb_total_replicas{model}  # Gauge
```

**Why Medium Priority**:
- Helps understand instance "fullness" (limited replica budget)
- High total replicas → instance is saturated
- Could influence routing of certain model requests

**Use Case**:
```promql
# Instance approaching replica capacity limit
sum(vllm:expert_replica_count) by (instance) > threshold
```

**Note**: This adds per-expert cardinality. For DeepSeek-R1 (256 experts), this is manageable but should be **optional** via flag `--enable-expert-metrics`.

---

### 4. **No Expert Load Distribution** 🔶 MEDIUM-LOW

**Problem**: Can't identify hot experts across instances (for global optimization).

**Missing Metrics**:
```python
# Tokens processed by each expert (aggregated by layer)
vllm:expert_tokens_total{model, layer, expert_id}  # Counter
```

**Why Medium-Low**:
- Useful for multi-instance analysis (which experts are hot globally?)
- Load balancer could route requests likely to hit hot experts to underutilized instances
- Requires workload-aware routing (advanced)

**Cardinality Concerns**:
- **High cardinality**: For DeepSeek-R1, this is 256 experts × 60 layers = 15,360 time series per instance
- **Mitigation**: Only enable with `--enable-expert-level-metrics` flag (default: OFF)
- **Alternative**: Aggregate to layer level only: `vllm:expert_tokens_by_layer_total{model, layer}`

**Use Case** (advanced):
```promql
# Identify globally hot experts across fleet
topk(10, sum(rate(vllm:expert_tokens_total[5m])) by (expert_id))

# Route traffic away from instances with hot expert replicas
vllm:expert_tokens_total{expert_id="42"} / avg(...) > 2
```

---

### 5. **Existing Metrics Are Sufficient** ✅

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

## Minimal Implementation Plan (3 Phases)

### Phase 1: Instance Health Signals (Week 1) - **MUST HAVE**

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
# Raw measurements (primary metrics)
vllm:eplb_avg_tokens_per_rank
vllm:eplb_max_tokens_per_rank
vllm:eplb_rebalance_events_total
vllm:eplb_rebalancing  # 0 or 1
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

---

### Phase 2: Capacity Signals (Week 2) - **SHOULD HAVE**

**Add to `EPLBStats`**:
```python
total_active_replicas: int = 0  # Sum of all expert replicas
num_unique_experts: int = 0     # Distinct experts with replicas
```

**Expose to Prometheus**:
```python
vllm:eplb_total_replicas{model}
vllm:eplb_unique_experts{model}
```

**Optional** (flag: `--enable-expert-metrics`):
```python
vllm:expert_replica_count{model, layer, expert_id}
```

**Load Balancer Usage**:
```python
# Avoid routing to instances near capacity
if eplb_total_replicas > capacity_threshold:
    weight *= 0.7
```

---

### Phase 3: Expert-Level Metrics (Week 3+) - **NICE TO HAVE**

**Only if `--enable-expert-level-metrics` flag set**:
```python
vllm:expert_tokens_total{model, layer, expert_id}
vllm:expert_load_imbalance{model, layer, expert_id}  # Per-expert skew
```

**Cardinality Control**:
- Sampling: only track top-K hottest experts (e.g., top 20)
- Aggregation: aggregate to layer level instead of per-expert
- Time-based: only export during high-load periods

**Load Balancer Usage** (advanced):
```python
# Route requests to instances with cold experts
# Requires workload modeling to predict expert usage
```

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

## Non-Goals (Out of Scope for MVP)

**From original requirements doc** - these are MoE/expert parallelism concepts that don't apply to vLLM's EPLB:
- ❌ Router queue metrics (vLLM doesn't have separate router component)
- ❌ Per-replica queue length (EPLB doesn't maintain per-replica queues)
- ❌ Expert processing duration histograms (too high cardinality, not actionable)
- ❌ Replica selection counters (routing is internal to worker, not exposed)
- ❌ Grafana dashboards (metrics first, visualization later)
- ❌ Alert rules (wait until metrics are deployed and baselined)

**Why**: These assume a different architecture (separate router/scheduler/worker services). vLLM integrates scheduling into the engine, so the relevant signals are at the instance level, not per-component.

---

## Success Criteria

Load balancer can answer:
1. ✅ **Is this vLLM instance healthy?** → Compute `1 - (avg_tokens / max_tokens)` to get imbalance ratio
2. ✅ **Does this instance have capacity?** → Check absolute load via `max_tokens_per_rank` and request queue
3. ✅ **Is this instance currently busy rebalancing?** → Check `eplb_rebalancing`
4. ✅ **How much headroom does this instance have?** → Check `num_requests_waiting` + `kv_cache_usage_perc` + `max_tokens_per_rank`
5. ✅ **Is this instance stable or thrashing?** → Check `rate(eplb_rebalance_events_total)`

---

## Files to Modify (Phase 1 Only)

1. **`vllm/v1/metrics/stats.py`**
   Add `EPLBStats` dataclass, add field to `SchedulerStats`

2. **`vllm/distributed/eplb/eplb_state.py`**
   Return stats from `step()` method

3. **`vllm/v1/core/sched/scheduler.py`**
   Populate `eplb_stats` in `make_stats()`

4. **`vllm/v1/metrics/loggers.py`**
   Add EPLB metrics to `PrometheusStatLogger.__init__()`

**Lines of code**: ~150 lines total

---

## Questions for Discussion

1. **Cardinality budget**: Are per-expert metrics (Phase 3) worth the cardinality cost for your use case?
   - DeepSeek-R1: 256 experts × 60 layers = 15K time series per instance
   - If you have 100 replicas, that's 1.5M time series just for expert metrics

2. **Rebalancing frequency**: How often is EPLB rebalancing in your workload?
   - Default is every 3000 steps - is this too frequent/infrequent?
   - Impacts how often `eplb_rebalancing` flag would be 1

3. **Load balancer sophistication**: Is your LB doing:
   - **Simple round-robin** → Only need Phase 1 (avg/max tokens for imbalance)
   - **Load-aware** → Phase 1 + existing `num_requests_waiting` + absolute token counts
   - **Workload-aware** → Phase 3 (expert-level metrics) might be useful

4. **Metrics export format**: Does your LB scrape Prometheus, or do you need push-based export (e.g., OTLP)?

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
