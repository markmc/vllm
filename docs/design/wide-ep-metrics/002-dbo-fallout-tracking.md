# RFC 2: DBO Fallout Tracking Metrics

## Motivation

Production WideEP deployments exhibit unpredictable throughput variance (10-50% degradation) with no clear diagnostic signal. Dual Batch Overlap (DBO) is a technique that splits batches into two micro-batches (ubatches) to overlap computation and communication, significantly improving throughput in Data-Parallel deployments. However, DBO is fragile and falls out silently when conditions aren't met—such as when the second ubatch would be empty (common at 256-token boundaries) or when DP ranks disagree on ubatching decisions.

Currently, there is no instrumentation to confirm whether DBO is engaged or why it disengages. Operators cannot distinguish between DBO working as expected, falling out due to batch size alignment issues, or failing due to coordination problems across DP ranks. This makes root-causing throughput degradation nearly impossible.

## Proposed Change

Add metrics to track DBO engagement and fallout reasons.

| Metric Name | Type | Description | Cardinality | Default |
|-------------|------|-------------|-------------|---------|
| `vllm:dbo_active` | Gauge | Whether DBO is currently engaged (1=active, 0=inactive) | 2 per instance (prefill/decode phases) | Enabled |
| `vllm:dbo_fallout_total` | Counter | Count of DBO fallout events by reason | ~3 per instance (reasons) | Enabled |
| `vllm:ubatch_token_count` | Histogram | Distribution of ubatch sizes in tokens | 2 per instance (first/second) | Enabled |

**Fallout reasons tracked**:
- `empty_second_ubatch` - Second ubatch would be empty after padding
- `coordination_failure` - DP ranks disagreed on ubatching decision
- `other` - Other reasons

**Total cardinality**: ~7 time series per instance

**Proposal origin**: `sallyom/dbo-eplb-stats` branch

**Label schema**: All metrics include `{model_name, engine}` labels. `dbo_active` adds `{phase}` (prefill/decode). `dbo_fallout_total` adds `{reason}`. `ubatch_token_count` adds `{ubatch_index}` (first/second).
