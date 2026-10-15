# Incremental iteration timing overhead

Measured on 2026-10-15.

The timing commit adds a small throughput cost in these GPU-busy 27B workloads:
point estimates range from **0.03% to 0.25% lower throughput**. The longer-prompt
workload is about **0.2% lower on both GPUs**, roughly 347.3 to 346.6 output
tokens/s. Two of the four 27B comparisons resolve a slowdown at the individual
95% level; two remain compatible with no change. None of these individual intervals
extends to a throughput loss larger than 0.45%. These are workload-specific intervals, not a universal or
simultaneous bound on production overhead.

The small-model throughput result is **inconclusive**: +0.32% on one GPU and
−0.34% on the other, with substantially wider intervals. It does not establish
either a speedup or zero timing overhead. In particular, this campaign does not
show that a measurable small-model penalty disappears on a larger model; the
small-model estimate is too noisy for that claim.

Negative changes below mean lower throughput with timing. Each cell contains
five paired repetitions; GPUs are not pooled.

| Workload | GPU 0: throughput change (95% CI) | GPU 1: throughput change (95% CI) |
| --- | ---: | ---: |
| Qwen3-0.6B, short | +0.32% (-0.85% to +1.49%) | -0.34% (-1.61% to +0.96%) |
| Qwen3.8-27B, matched short | -0.25% (-0.45% to -0.05%) | -0.03% (-0.26% to +0.21%) |
| Qwen3.8-27B, longer prompts | -0.19% (-0.45% to +0.07%) | -0.20% (-0.37% to -0.02%) |

For a deployment focused on keeping this GPU busy, the measured incremental
price is a few tenths of a percent, rather than a multi-percent regression.
That can be a reasonable operability tradeoff if the timings help diagnose
serving problems. This benchmark measures the price, not the value of that
visibility. Maintainers should make that judgement from the expected diagnostic
benefit and acceptable performance cost; the measurements do not require
rejecting the feature simply because some overhead is measurable.

This comparison keeps count collection enabled in both revisions, so it does
not decide whether collection as a whole should become enabled by default.
It also does not establish the worst case for latency-sensitive services.

The measured campaign completed in **72.1 minutes**, within the 105-minute
execution limit. All 40 server blocks and 60 case measurements passed: 81,920
requests, 20,480,000 generated tokens, exact input/output totals, zero failed
requests, zero preemptions, stable source identities, and unchanged measurement
scripts. Each measured window lasted 65.4–76.8 seconds. A subsequent wait for
GPU-tool approval delayed final validation after the measured campaign had
finished; the total interaction took approximately two hours.

## What this comparison measures

Compare `a33a88263f41d0e99dc0e4568f9df5e2a7833ff0` (baseline) with
`5cdcb87fece64ccb069cb8c8b9a4002b3c18dec3` (candidate), with iteration-details
collection and ordinary statistics **enabled in both**. Async scheduling is
enabled throughout. This is a sensible comparison for the incremental serving
cost of the timing commit.

It measures the complete change: CUDA event recording and retrieval, association
of timings with iteration counts, delayed delivery and polling, and preparation
of logging arguments. It also includes removal of the baseline's CPU wait timer.
It does not isolate CUDA event calls alone, measure the original count-collection
cost, or measure timing with collection disabled. Console logging uses WARNING,
so writing a line for every iteration is outside this comparison; argument
preparation performed before a suppressed logging call remains included.

## Why these workloads

The small model is a sensitivity control. Its short GPU execution time gives
CPU-side bookkeeping more opportunity to affect throughput. This helps expose
cost that a larger model might hide, but a small model is not automatically
CPU-bound: its GPU was also busy in preparation. Its percentage slowdown should
not stand in for the production cost that should inform an enablement decision.

Qwen3.8-27B supplies a more relevant check for deployments trying to get useful
work out of an expensive GPU. We run it with the same short request shape and
concurrency as the small model, then with longer prompts as a second GPU-busy
workload. The model architectures differ, so differences cannot be attributed
solely to parameter count. This is a validated production proxy on this host,
not a survey of enterprise models or distributed deployments.

| Model / workload | Input / output tokens | Concurrency | Requests | Warmup requests |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B / short | 32 / 256 | 64 | 3,584 | 128 |
| Qwen3.8-27B / matched short | 32 / 256 | 64 | 320 | 64 |
| Qwen3.8-27B / longer prompts | 1,024 / 128 | 64 | 192 | 64 |

Each A100 80GB PCIe runs both revisions and both models, with five alternating
pairs per workload. GPUs run independently with disjoint CPU allocations and
separate ports. Each 27B server runs both request shapes before shutdown, keeping
startup within the recurring campaign budget. Concurrency 64 was the highest
throughput point tested in preparation; it is not a proven global optimum.

Both models use BF16, TP=1, a 2,048-token context limit, 64 maximum sequences and
an 8,192-token scheduling budget. Prefix caching, speculative decoding and
quantization are disabled. Compilation and CUDA graphs remain enabled. The 27B
model loads text-only. Requests use fixed random lengths and ignore EOS. Full
model revision pins, CPU affinity and memory limits are saved in the plan.

The primary metric is output-token throughput. Confidence intervals use paired
per-run log ratios and Student's t distribution, with each GPU analyzed
separately. Requests and individual token gaps are not independent repetitions.
Five pairs give a useful first estimate, with the resulting intervals stating
how precisely this campaign can resolve a small effect. Latency percentiles are
supporting observations, especially because tails are sensitive to request
batching and this small number of repetitions.

## GPU activity and supporting latency results

During the middle 80% of every 27B measurement, median GPU activity was 100%
and median power was 296.9–300.6 W. Per-run median SM clocks ranged from 1,125
to 1,305 MHz across the two workload shapes. The client used under 0.028 CPU
cores. Small-model activity was 92–99%, with 1,410 MHz clocks and about 0.36
client CPU cores. This supports a GPU-busy 27B comparison with ample client
headroom; it does not demonstrate a CPU scheduling bottleneck in the small model
or a globally optimal deployment configuration.

The 27B median inter-token latency increased by 0.03–0.18% across the four
comparisons. Short-prompt p99 inter-token latency increased by 0.33–0.74%; the
GPU 0 interval resolves an increase of 0.32–1.16%, while GPU 1 includes zero.
Other 27B tail intervals include zero. The longer-prompt workload already has
substantial batching delays: roughly 6 seconds median time to first token and
2 seconds p99 inter-token latency. This is a throughput-oriented operating
point, not evidence for a latency-sensitive service or an SLO-tuned setup.
Small-model tails vary too much to support a clear timing-cost conclusion.

## Evidence and reproduction

See [PREPARATION.md](PREPARATION.md) for model validation, concurrency selection
and the runtime forecast, and [README.md](README.md#repeatable-serving-campaign)
for the campaign commands. The inference environment and 275 installed artifacts
are shared across revisions; artifact SHA-256 hashes were verified when creating
the frozen source copies. The main implementation checkout is not used by the
running servers.

The first launch failed argument parsing before either server started because
the prepared serving wrapper had been left in the rebase stash. The wrapper and
its documentation were recovered without applying or dropping the rest of the
stash. That launch produced no performance observations and is excluded.

Raw evidence is retained locally under
`.cache/iteration_stats/timing-5cdcb87-20261015/`: per-run results and commands,
source identities, Prometheus snapshots, one-second GPU telemetry, client CPU
traces, environment versions and artifact hashes. The campaign execution limit
is 105 minutes, leaving analysis time within the approximate two-hour budget.

The compact audited results, including every paired ratio, absolute throughput,
latency percentiles, and per-run GPU/client summaries, are saved in
[iteration-timing.json](results/2026-10-15/iteration-timing.json).
[Source and artifact hashes](results/2026-10-15/iteration-timing-sources.json) and
[environment and script identities](results/2026-10-15/iteration-timing-environment.json)
allow the comparison to be reproduced.

The separate [untimed delivery check](results/2026-10-15/iteration-timing-delivery.json)
received 130 unique iteration records with valid, finite GPU durations and the
`model_step_cuda` scope, including prefill and decode counts. It passed with no
preemptions and exit status zero. Its offline helper uses FP16 and instruments
the frontend, so its latency is excluded from the serving results. At process
exit it emitted an MPClient engine-core cleanup warning; no such warning or
traceback appeared in the measured serving logs. Both frozen snapshots' 275
installed-artifact hashes were checked again after the campaign and matched.
