# RFC 3: Per-Layer EPLB Metrics

## Motivation

RFC 1 provides instance-level EPLB metrics sufficient for load balancing decisions. However, when debugging performance issues in production, operators need to identify which specific MoE layers are experiencing imbalance. Instance-level aggregates show that an instance is struggling but not which layer is the bottleneck. Different layers can have dramatically different expert selection patterns based on the workload, and layer-specific visibility is critical for root-causing issues like expert hotspots or identifying whether the problem is limited to early vs. late layers.

This proposal extends RFC 1 metrics with per-layer granularity. While the cardinality increase (~4 to ~180 time series) is moderate, it's 45x higher than RFC 1, justifying opt-in behavior to avoid unnecessary metric volume in deployments that don't need layer-level debugging.

## Proposed Change

Add per-layer variants of RFC 1 EPLB metrics, disabled by default.

| Metric Name | Type | Description | Cardinality | Default |
|-------------|------|-------------|-------------|---------|
| `vllm:eplb_avg_tokens_per_rank` | Gauge | Average tokens per EP rank in current interval | ~60 per instance (1 per layer) | **Opt-in** |
| `vllm:eplb_max_tokens_per_rank` | Gauge | Maximum tokens across EP ranks in current interval | ~60 per instance (1 per layer) | **Opt-in** |
| `vllm:eplb_balancedness_ratio` | Gauge | Load balancedness ratio (avg/max) | ~60 per instance (1 per layer) | **Opt-in** |
| `vllm:eplb_rebalancing` | Gauge | Whether EPLB is currently rebalancing (0 or 1) | 1 per instance | **Opt-in** |
| `vllm:eplb_rearrangements_total` | Counter | Total number of EPLB rebalancing operations | 1 per instance | **Opt-in** |
| `vllm:eplb_rearrangement_duration_seconds` | Histogram | Duration of rebalancing operations | 1 per instance | **Opt-in** |

**Total cardinality**: ~183 time series per instance (for 60-layer models like DeepSeek-R1)

**Proposal origin**: `sallyom/dbo-eplb-stats` branch

**Label schema**: All metrics include `{model_name, engine}` labels. Per-layer metrics add `{layer}` label.

**Relationship to RFC 1**: This proposal supersedes RFC 1 by adding layer granularity. If adopted, RFC 1's instance-level metrics would remain enabled by default, while these per-layer variants would be opt-in. Alternatively, these could replace RFC 1 entirely with instance-level aggregates computed in PromQL when needed.
