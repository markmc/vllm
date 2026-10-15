# Repeatable serving campaign preparation

Prepared on 2026-10-15 for two independent A100 80GB TP=1 experiments.
Preparation trials select workloads and estimate runtime; they are not paired
overhead measurements and do not establish a new enablement recommendation.

## Comparison and models

The frozen baseline is `a33a88263f`, containing count collection. The frozen
candidate is `5cdcb87fec`, adding nonblocking GPU iteration timing. Both enable
iteration details with ordinary statistics enabled and WARNING logging. The
next campaign therefore measures the incremental timing change, including its
delivery. It does not isolate the original count-collection cost.

Use a different, explicitly recorded baseline or enablement combination when
benchmarking another change. The runner checks source revisions against the
plan and rejects dirty vLLM source trees.

| Model | Immutable revision | Role |
| --- | --- | --- |
| Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | Small-model control. |
| Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Larger-model control and GPU-busy serving workload. |

The [published recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) has no A100
selection. This preparation validates a host-specific configuration rather
than claiming that the published H200 command was verified on an A100.

The 27B configuration uses BF16, TP=1, text-only loading, a 2,048-token context
limit, 64 maximum running sequences, an 8,192-token scheduling budget and 90%
GPU-memory utilization. Compilation and CUDA graphs remain enabled. There is
no quantization, speculative decoding or prefix caching. Small-model controls
use the same dtype, context, sequence and token limits, with 60% GPU-memory
utilization. Clients use fixed random lengths and ignore EOS.

## Checkpoint storage and frozen sources

The root filesystem had only 27 GB free. The complete 55.6 GB checkpoint was
downloaded to the host cache `/dev/shm/iteration-stats-hf` in about seven minutes.
This is a RAM-backed cache and does not survive reboot. The host had 238 GiB
available memory and 126 GiB free shared memory before the download. No existing
files were deleted and the inference environment was not changed.

The cache is visible outside the GPU sandbox, as required for serving. Its
model revision is pinned in [campaign.json.example](campaign.json.example).
To restore the checkpoint after a reboot, provide sufficient storage and run:

```bash
HF_HUB_CACHE=/dev/shm/iteration-stats-hf \
HF_XET_CACHE=/dev/shm/iteration-stats-xet \
HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
  .venv/bin/hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

Download the whole snapshot: the installed Hugging Face client rejects an
offline snapshot missing README, license or Git metadata even when all weights
are present. Download and initial compatibility work are one-time preparation,
outside the recurring campaign budget.

The preparation used revisions `3d478485ba` and `b43a653a80`; their `vllm/`
source trees are identical to `a33a88263f` and `5cdcb87fec`, respectively.
Frozen source copies at the latter exact commits, with 275 identical installed
artifacts verified by SHA-256, are in
`/tmp/iteration-timing-sources-5cdcb87-20261015/{base,patch}`. `snapshot.json` in that
directory records full revisions, source identity and artifact hashes. Recreate
them for subsequent changes with `snapshot.py`; update the plan's revision pins.

## Selection and runtime

Select concurrency **64** for all three workloads. Both frozen revisions
completed the isolated 32/64 sweep with exact input/output totals and no
preemption. The corrected preparation results are:

| Qwen3.8-27B workload | Concurrency | GPU 0 candidate output tok/s | GPU 1 baseline output tok/s |
| --- | ---: | ---: | ---: |
| 32 input / 256 output | 32 | 668.4 | 667.7 |
| 32 input / 256 output | 64 | 1,068.3 | 1,066.2 |
| 1,024 input / 128 output | 32 | 289.1 | 289.2 |
| 1,024 input / 128 output | 64 | 345.8 | 346.5 |

These are calibration trials on different GPUs, not overhead comparisons.
Concurrency 64 has the highest tested throughput. Throughput is still rising
between 32 and 64, so this is not a demonstrated global optimum or a measured
plateau. It is a validated GPU-busy operating point within the selected engine
sequence and memory limits.

Both 27B workloads had median GPU activity of 100% and approximately 297-301 W
during the middle 80% of the measured request window. The client used less than
0.03 CPU cores there. The small-model concurrency-64 pilot delivered about
14,048 output tok/s with 99% median GPU activity and about 0.38 client CPU cores.
That small-model pilot used a 128-sequence server limit; the repeated plan fixes
both models at 64. Its runtime calibration is therefore approximate. A small
model is a sensitivity control, not evidence that the workload is CPU-bound.

Warm-cache 27B startup took **69.1 seconds** on each GPU. The corrected four-case
blocks also measure client startup and warmup, which are included in the forecast.
The first model initialization required roughly four minutes including initial
compilation; recurring runs should retain the compilation cache.

| Repeated workload | Concurrency | Requests per run | Warmup requests | Estimated measured duration |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B, 32 / 256 | 64 | 3,584 | 128 | 65 seconds |
| Qwen3.8-27B, 32 / 256 | 64 | 320 | 64 | 77 seconds |
| Qwen3.8-27B, 1,024 / 128 | 64 | 192 | 64 | 71 seconds |

Scaling the measured request rates while retaining observed startup and warmup
costs predicts approximately **80 minutes per GPU**, with both GPUs running
concurrently. Allow roughly 90-105 minutes including cold-cache variation and
analysis. This preparation forecast is now supplemented by the completed timing campaign:
its 40 server blocks took 72.1 minutes; see [TIMING_RESULTS.md](TIMING_RESULTS.md). The
plan saves the calculation inputs, including a conservative 15-second allowance
for small-model wrapper startup before server launch.

The corrected concurrency sweep and warm-cache restart measurements are saved
under `.cache/iteration_stats/qwen38-preparation-20261015/`. Both revisions must
complete requests with exact input/output totals and no preemption before a
workload is selected. Compare concurrency 32 and 64 using the same engine
sequence limit, and inspect GPU activity and benchmark-client CPU usage during
the measured request window.

The final request counts are calibrated from preparation throughput, then frozen
across configurations and repetitions. The repeated plan has three workloads:
matched short prompts on each model and 1,024-token prompts on the 27B model.
There are five pairs per GPU, twenty server starts per GPU, and both GPUs run
concurrently. The 27B server handles both workload shapes before shutdown.

The runner has a 110-minute execution limit to leave time for validation and
confidence-interval analysis within the approximate two-hour budget. Budget
expiry produces an incomplete campaign error, not a complete result. The
request counts and measured runtime forecast are recorded in the plan.

## Validation and excluded trials

Initial concurrent 27B trial wrappers accidentally used the same HTTP port.
GPU telemetry exposed the routing problem. All their measurements are excluded
under `invalid-port-collision/`, with the reason saved beside them.

The corrected trials use ports 18090 and 18091. The reusable runner rejects
duplicate ports and GPU assignments. Each serving wrapper also verifies that
the listening socket belongs to its own server process or a descendant before
accepting health readiness, preventing a healthy unrelated server from
silently satisfying the check.

Source, command, hardware and workload metadata, per-case results, Prometheus
snapshots, client CPU traces and one-second GPU telemetry remain in the local
preparation archive. Failed and short trials are calibration evidence only;
they are not pooled into the repeated campaign. Serving tail metrics remain
available, but the primary comparison is output-token throughput with paired
confidence intervals. This single-node workload does not establish overhead
for all production models, contexts or distributed deployments.

See [README.md](README.md#repeatable-serving-campaign) for dry-run, execution and
analysis commands. The repeated timing campaign is documented in [TIMING_RESULTS.md](TIMING_RESULTS.md).
