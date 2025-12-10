# RFC 5: Per-Expert Metrics with Cardinality Controls

## Motivation

In rare production incidents, aggregate metrics (RFC 1-4) may show that an instance is struggling but not reveal which specific expert is the bottleneck. For example, a single "hot" expert receiving 10x more selections than others can degrade throughput, but this is invisible in per-rank or per-layer aggregates. Per-expert metrics enable operators to identify exactly which expert IDs are causing hotspots during active incidents.

However, per-expert metrics create extreme cardinality: 256 experts × 60 layers = 15,360 time series per instance. With 100 replicas, this creates 1.5M time series cluster-wide. This cardinality explosion is only justified during active debugging, not continuous monitoring. This proposal includes safety controls to prevent accidental metric explosions while still enabling deep debugging when needed.

## Proposed Change

Add opt-in per-expert metrics with automatic safety controls.

| Metric Name | Type | Description | Cardinality | Default |
|-------------|------|-------------|-------------|---------|
| `vllm:moe_expert_selection_counter` | Counter | Cumulative count of expert selections per expert | ~15,360 per instance (layers × experts) | **Disabled** |
| `vllm:expert_load_per_expert_tokens_DEBUG` | Gauge | Current token load per expert (with `_DEBUG` suffix warning) | ~15,360 per instance (layers × experts) | **Disabled** |

**Total cardinality**: ~15,360 time series per instance when enabled (256 experts × 60 layers for DeepSeek-R1)

**Proposal origins**:
- `moe_expert_selection_counter`: PR #19915 and PR #27105
- `expert_load_per_expert_tokens_DEBUG`: `sallyom/dbo-eplb-stats` branch

**Label schema**: `{model_name, engine, layer, expert}` (or `expert_id`)

**Safety controls**:
1. **Feature flag required**: Must explicitly enable via `--enable-expert-metrics` or environment variable
2. **Auto-disable timer**: Metrics automatically turn off after configurable duration (e.g., 5-30 minutes) to prevent indefinite high-cardinality export
3. **`_DEBUG` suffix**: Metric name includes warning suffix to signal high cardinality
4. **Documentation warnings**: Configuration docs clearly state cardinality impact

**Alternative approach (future enhancement)**: Instead of exporting all experts, export only top-K hottest experts (e.g., top 100) to cap cardinality at ~6,000 time series while still identifying bottlenecks.

**Use case**: Incident debugging only. Not intended for continuous monitoring.
