# RFC 1: Instance-Level EPLB Metrics for Load Balancing

## Motivation

In replicated vLLM deployments behind a load balancer, the load balancer has no visibility into EPLB (Expert-Parallel Load Balancing) state when making routing decisions. EPLB dynamically replicates heavily-used experts across EP ranks to balance load, and instances experiencing high expert load imbalance or active rebalancing operations have degraded capacity. Without metrics exposing this state, load balancers cannot avoid routing traffic to struggling instances, leading to suboptimal performance and increased tail latencies.

EPLB already tracks and logs this information to stdout when `--eplb-log-balancedness` is enabled, but logs are ephemeral and not queryable by load balancers. This proposal exports the same data that's currently logged as Prometheus metrics, enabling intelligent load balancing decisions based on instance health.

## Proposed Change

Export instance-level EPLB metrics to Prometheus. All metrics represent data already computed and logged by EPLB.

| Metric Name | Type | Description | Cardinality | Default |
|-------------|------|-------------|-------------|---------|
| `vllm:eplb_avg_tokens_per_rank` | Gauge | Average tokens per EP rank in current interval | 1 per instance | Enabled |
| `vllm:eplb_max_tokens_per_rank` | Gauge | Maximum tokens across EP ranks in current interval | 1 per instance | Enabled |
| `vllm:eplb_rebalancing` | Gauge | Whether EPLB is currently rebalancing (0 or 1) | 1 per instance | Enabled |
| `vllm:eplb_rebalance_events_total` | Counter | Total number of EPLB rebalancing operations | 1 per instance | Enabled |

**Total cardinality**: ~4 time series per instance

**Proposal origin**: `epblb-metrics-claude` branch

**Label schema**: All metrics include `{model_name, engine}` labels consistent with existing vLLM metrics.
