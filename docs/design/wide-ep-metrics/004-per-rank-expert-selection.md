# RFC 4: Per-Rank Expert Selection Distribution

## Motivation

EPLB metrics (RFC 1 and RFC 3) show aggregate token load across EP ranks but don't reveal the underlying cause: which ranks are receiving disproportionate expert selections. When EPLB shows high imbalance, operators need to understand whether the problem is concentrated on specific ranks or distributed unevenly across all ranks. This information helps distinguish between EPLB configuration issues (e.g., expert replication not aggressive enough) versus workload characteristics (e.g., certain prompts consistently routing to the same expert subset).

Per-rank expert selection counters provide this visibility with low cardinality (60 layers × 4 ranks = 240 time series for typical configurations). This is only 33% more than RFC 3's per-layer metrics, making it reasonable to consider enabling by default or as an opt-in companion to RFC 3.

## Proposed Change

Add per-rank expert selection counters to track which EP ranks are processing the most tokens.

| Metric Name | Type | Description | Cardinality | Default |
|-------------|------|-------------|-------------|---------|
| `vllm:moe_per_rank_expert_selection_counter` | Counter | Cumulative count of expert selections per rank | ~240 per instance (layers × ranks) | **TBD** |

**Total cardinality**: ~240 time series per instance (60 layers × 4 EP ranks for DeepSeek-R1)

**Proposal origin**: PR #27105 (`patryk/expert-usage-histogram` branch)

**Label schema**: `{model_name, engine, layer, rank}`

**Default behavior**: To be determined. Low cardinality suggests this could be enabled by default, but the incremental value over RFC 3 may justify opt-in behavior to minimize metric volume for deployments that don't need rank-level granularity.

**Implementation note**: Requires Triton kernel `collect_expert_usage_histogram()` from PR #27105 for efficient histogram collection. When EPLB is enabled, physical experts are remapped to logical experts before aggregation.
