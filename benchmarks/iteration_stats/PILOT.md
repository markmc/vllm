# Collection benchmark preparation and pilot

Date: 2026-10-13. These are setup and calibration trials, not the repeated
measurements required to choose default enablement.

Pilot revision: `5325ad641818b0250846ab904bf9642520e33818`.
Current collector revision: `88cc803f0417c3d93a4e0da28978ef0e8bcd2e2a`.
Both have the same vLLM source tree: `31899508a34386447e8d076da1b0f736c286a87c`.
Base revision: `c83935b3008553c72b1bad09f842b4c99b7fa47f`.

The pilot ran Qwen3-0.6B FP16, TP=1, 32 input tokens and 256 output tokens
on both A100 80GB GPUs. It used separate physical CPU cores, ordinary stats
enabled, WARNING logging, no prefix caching, no detokenization, and compilation
with FULL_AND_PIECEWISE CUDA graphs. Startup and three warmup batches were
excluded. Each configuration measured three batches in one paired repetition.
All 16 measured configurations completed without preemption or code changes.

## Exploratory C/B results

B is the collector patch with collection disabled; C enables collection.
Latency changes below are individual paired point estimates. There is only
one paired repetition per case, so confidence intervals are unavailable.

| GPU | Batch | Async | B latency (s) | C latency (s) | C/B latency change |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | on | 0.538414 | 0.537850 | -0.105% |
| 0 | 1 | off | 0.812335 | 0.810270 | -0.254% |
| 0 | 512 | on | 3.654313 | 3.679649 | +0.693% |
| 0 | 512 | off | 5.850240 | 5.889878 | +0.678% |
| 1 | 1 | on | 0.538286 | 0.538352 | +0.012% |
| 1 | 1 | off | 0.800005 | 0.802813 | +0.351% |
| 1 | 512 | on | 3.639503 | 3.671921 | +0.891% |
| 1 | 512 | off | 5.842724 | 5.876398 | +0.576% |

The largest batch showed roughly 0.6–0.9% higher latency with collection
enabled. This justifies prioritizing batch 512 in the repeated comparisons;
these short, unreversed pilot pairs do not establish the size or consistency
of that effect. Batch 32 and 128 still need their primary pilot sweeps.

## Calibration for 90-second measurements

| GPU | Batch | Async | Suggested measured batches |
| --- | --- | --- | --- |
| 0 | 1 | on | 168 |
| 0 | 1 | off | 111 |
| 0 | 512 | on | 25 |
| 0 | 512 | off | 16 |
| 1 | 1 | on | 168 |
| 1 | 1 | off | 113 |
| 1 | 512 | on | 25 |
| 1 | 512 | off | 16 |

Use the same measured-batch count for every configuration in a pair. Confirm
actual measured duration and warmup stability before accepting a repetition.

## Delivery and serving setup

Untimed checks on the base revision verified disabled/enabled gating at
WARNING level. Checks on the collector verified frontend delivery with async
scheduling enabled and disabled, using batch 32. Each check covered one warmup
and one generation: 2,048 prefill input tokens and 16,320 decode input tokens
reached the frontend, matching 64 requests of 32 inputs and 255 decode inputs.
Neither check found duplicate iteration indices or preemption.

The serving helper uses `--random-range-ratio 0` in this revision to keep
input/output lengths fixed. A smoke run completed all 128 requests with
4,096 input tokens, 32,768 output tokens and zero preemptions. The initial
revision guard rejected a commit-message amendment despite identical code;
the helper now compares source trees and records start/end commit IDs.

Qwen3-8B is cached at `b968826d9c46dd6066d109eabc6255188de91218`.
The small model uses `c1899de289a04d12100db370d81485cdf75e47ca`.

## Saved evidence and remaining measurements

Scripts and complete run instructions: [README.md](README.md).
Raw pilot results, exact commands, logs and source/artifact provenance:
[preparation archive](../../.cache/iteration_stats/5325ad64-preparation/).

The remaining benchmark work is the full 1/32/128/512 primary pilot, request
churn and 8B control, timed A/B/C comparisons, unchanged-versus-unchanged noise,
five or more 60–120-second paired repetitions on both GPUs, and repeated
serving confirmation with median/p99 TTFT and ITL. Use the measured evidence
to choose sensitive cases and extend repetition where confidence intervals
cannot exclude a 1% regression. These pilot results support no default
collection-enablement conclusion.
