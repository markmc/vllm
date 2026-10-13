# Scheduled iteration collection benchmarks

See [RESULTS.md](RESULTS.md) for the completed campaign and enablement recommendation.

Preparation for the phase-1 benchmark plan in
[scheduled_iteration_stats.md](../../docs/design/scheduled_iteration_stats.md).
These tools measure collection and its existing delivery path. Keep histogram
additions and worker timing in separate stages.

`latency.py` delegates measurement to `vllm.benchmarks.latency.main`. It explicitly
checks ordinary `log_stats`, disables detokenization and prefix caching, seeds
pretokenized inputs after engine initialization, and uses `WARNING` logging.
`EngineArgs.disable_log_stats=False` is passed through `LLM.from_engine_args`;
the CLI has no `--no-disable-log-stats` option. Compilation and graphs retain
their ordinary enabled configuration. No delivery hooks run in measured trials.

`run.py` saves raw batch latencies, output-token throughput, effective settings,
commands, source revisions/diffs, package versions, native artifact hashes and
hardware information. Runs fail on preemption or source changes. Use a new output
directory for each invocation. Model files must already be cached at the supplied
immutable revision. GPU access on this host requires execution outside the Codex
sandbox.

## Sources and hardware

The committed collector is `88cc803f0417c3d93a4e0da28978ef0e8bcd2e2a`.
The initial pilot recorded `5325ad641818b0250846ab904bf9642520e33818`;
the amendment preserves the same vLLM source tree.
Its base is `c83935b3008553c72b1bad09f842b4c99b7fa47f`.
The cached Qwen3-0.6B model revision is
`c1899de289a04d12100db370d81485cdf75e47ca`.
Qwen3-8B is also cached at
`b968826d9c46dd6066d109eabc6255188de91218`.

GPU 0 and GPU 1 are A100 80GB PCIe devices, both on NUMA node 0.
GPU 2 is a separate 40GB device and is outside the planned comparisons.
CPU 56 is the SMT sibling of CPU 0; use one thread per physical core.
Suggested disjoint allocation:

| Process | GPU | Physical CPU threads |
| --- | --- | --- |
| Engine and offline frontend | 0 | `0,2,4,6,8,10` |
| Engine and offline frontend | 1 | `12,14,16,18,20,22` |
| Serving client | 0 | `24,26` |
| Serving client | 1 | `28,30` |

Check GPU availability and host load before starting a comparison. Keep the same
allocation and GPU UUID throughout each set of paired measurements. Avoid running
two benchmark processes on the same GPU.

The main checkout can be used directly after committing the change. For the
base revision, or to protect trials from further edits, create isolated copies
with identical installed artifacts:

```bash
.venv/bin/python benchmarks/iteration_stats/snapshot.py \
  --base-revision c83935b3008553c72b1bad09f842b4c99b7fa47f \
  --output-dir /tmp/iteration-stats-sources
```

This creates `base` and `patch` copies and records artifact checksums. The patch
copy uses current HEAD plus any current vLLM diff; the base copy uses the specified
revision. Copies share Git objects with the original repository and use the
existing Python environment. Keep the original repository available while using
them. Recreate copies after code or native-artifact changes.

## Pilot and delivery verification

Run a separate untimed delivery check on each scheduling mode, keeping logging
at `WARNING`. It records frontend iteration payloads, checks collection gating,
and rejects duplicate iteration indices. Its latency results are excluded from
performance analysis.

```bash
.venv/bin/python benchmarks/iteration_stats/run.py \
  --repo /tmp/iteration-stats-sources/patch \
  --output-dir /tmp/iteration-stats-delivery \
  --gpu 0 --cpu-list 0,2,4,6,8,10 \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --batch-sizes 32 --async-modes on off --configs B C \
  --repeats 1 --num-iters 1 --warmup 1 --verify-delivery
```

Start the primary pilot with all four batch sizes. Repeat the invocation on
GPU 1 using its own CPU allocation and output directory.

```bash
.venv/bin/python benchmarks/iteration_stats/run.py \
  --repo /tmp/iteration-stats-sources/patch \
  --base-repo /tmp/iteration-stats-sources/base \
  --output-dir /tmp/iteration-stats-pilot-gpu0 \
  --gpu 0 --cpu-list 0,2,4,6,8,10 \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --batch-sizes 1 32 128 512 --async-modes on off \
  --configs A B C --repeats 1 --num-iters 5 --warmup 10
```

Use pilot mean latency to choose `--num-iters`: approximately
`ceil(90 / mean_batch_latency_seconds)`. Run each selected batch size separately
when iteration counts differ. Confirm each run actually measures 60–120 seconds,
and inspect raw samples for warmup drift. Startup and graph capture are excluded
by the existing benchmark. Warm every measured batch shape before measurement.

For repeated comparisons use `--configs A B C B-repeat --repeats 5`.
Configuration order alternates and rotates across repetitions. `B-repeat` is
identical to B and establishes unchanged-versus-unchanged noise. Optional
`A-enabled` measures legacy enabled collection.

Request churn uses `--input-len 32 --output-len 1 --batch-sizes 32 128`.
The representative control uses `--model Qwen/Qwen3-8B`, its immutable cached
revision, `--input-len 1024 --output-len 128 --batch-sizes 32 128`.
The 16,384-token budget chunks larger control prefills; it remains identical
across configurations. Model evaluation is outside this performance-only harness.

## Serving confirmation

`serve.py` starts a local server, waits for readiness, runs `vllm bench serve`
with a separately pinned client, saves detailed results and Prometheus snapshots,
then stops its own server process group. It uses fixed random lengths,
`--ignore-eos`, unlimited arrival rate, bounded concurrency, and warmup requests.
The server uses `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120`, exceeding the benchmark
client's 60-second connection retention to avoid stale connection reuse after
warmup.
The client tokenizer comes from the exact cached model revision. It saves median
and p99 TTFT and ITL, and rejects incomplete runs and preemptions.

```bash
.venv/bin/python benchmarks/iteration_stats/serve.py \
  --repo /tmp/iteration-stats-sources/patch \
  --output-dir /tmp/iteration-stats-serving-C \
  --gpu 0 --server-cpus 0,2,4,6,8,10 --client-cpus 24,26 \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --collection on --async-mode on --concurrency 32 --num-prompts 1024
```

Calibrate `--num-prompts` to a 60–120-second benchmark duration. Repeat paired
off/on trials in alternating order on both GPUs, and also compare the base
revision where needed. For concurrency 512, increase warmups if transient admission
and draining shapes still trigger compilation during measurement.

## Analysis

Use an isolated analysis environment so dependency installation does not change
the inference environment:

```bash
uv venv /tmp/iteration-stats-analysis/.venv --python .venv/bin/python
uv pip install --python /tmp/iteration-stats-analysis/.venv/bin/python scipy
/tmp/iteration-stats-analysis/.venv/bin/python benchmarks/iteration_stats/analyze.py \
  /tmp/iteration-stats-pilot-gpu0 /tmp/iteration-stats-pilot-gpu1 \
  --output /tmp/iteration-stats-summary.json
```

The analyzer compares paired per-run mean latencies within each GPU. It uses
Student-t confidence intervals on log ratios, with repetitions as independent
samples. Fixed-length output throughput is the reciprocal of mean latency;
it provides the same evidence. Different source snapshots cannot be pooled.
Runs with fewer than five pairs or less than 60 measured seconds are marked as
pilots. Inspect raw distributions, unchanged-versus-unchanged noise, and serving
tails before drawing a default-enablement conclusion. A confidence interval
crossing zero does not establish acceptable overhead. The provisional target is
to exclude a 1% regression in the sensitive workload. This target informs the
decision alongside the measured cost and operational value of the metrics;
it is not a universal observability budget. See [RESULTS.md](RESULTS.md) for the
completed measurements and interpretation.

## Warmed sessions and complete campaigns

`matrix.py` runs several workload shapes in one engine session per configuration.
Each shape receives its own warmup and stock latency measurement. The first
configuration calibrates its sample count from warmup to `--seconds`; the same
count is then used throughout its paired comparison. Source, artifact, command,
hardware and effective-settings records remain separate for every shape.
GPU clocks, temperature, power and utilization are sampled every five seconds.

Supply a JSON list of objects containing `name`, `batch_size`, `input_len`, and
`output_len`. For example:

```json
[
  {"name": "decode1", "batch_size": 1, "input_len": 32, "output_len": 256},
  {"name": "decode512", "batch_size": 512, "input_len": 32, "output_len": 256}
]
```

```bash
.venv/bin/python benchmarks/iteration_stats/matrix.py \
  --repo . --base-repo /tmp/iteration-stats-sources/base \
  --output-dir /tmp/iteration-stats-campaign/sensitive-gpu0 \
  --cases-file /tmp/iteration-stats-campaign/sensitive.json \
  --gpu 0 --cpu-list 0,2,4,6,8,10 \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --configs A B C B-repeat --repeats 5 --warmup 8 --seconds 75
```

`campaign.py` waits for a successful sensitive sweep before running the 8B
control, request churn and paired serving confirmation on that GPU. It expects
`control.json` and `churn.json` in the campaign root. Each GPU has its own journal.
Serving uses concurrency 512 in both scheduling modes, five alternating pairs,
and a pilot to select a fixed request count for approximately 75 measured seconds.
For additional repetitions of an unresolved synchronous batch-512 comparison,
provide `sensitive-confirm.json` containing that shape and use
`campaign.py --confirm-sensitive`. This waits until the GPU's main campaign
finishes, then runs five additional A/C pairs. The analyzer pools these with
the original pairs only when implementation identities and workload settings
match. Serving also saves a one-second client CPU trace to help identify a
client throughput limit.
`--serving-only` restarts the serving stages without repeating completed offline
matrices. Archive earlier serving directories and journals before restarting so
different protocols cannot be pooled.

`cpu.py` diagnoses the actual synchronous `Scheduler.schedule()` path using
fixed decode context and setup outside timing. It measures collector CPU cost
without model execution, serialization or frontend delivery. This helps explain
GPU measurements; its percentage cost is not an end-to-end serving regression.

`summarize.py` combines completed repeated latency matrices and serving pairs.
Short screening trials remain separate. `audit.py` checks configuration, source
stability, identical native artifacts and the measurement harness, and reports
within-run first/last-third latency drift. Use `--partial` on either script to
inspect progress; final analysis must omit it.

```bash
uv pip install --python /tmp/iteration-stats-analysis/.venv/bin/python \
  scipy matplotlib
/tmp/iteration-stats-analysis/.venv/bin/python \
  benchmarks/iteration_stats/summarize.py \
  --root /tmp/iteration-stats-campaign \
  --output-dir /tmp/iteration-stats-campaign/summary
.venv/bin/python benchmarks/iteration_stats/audit.py \
  --root /tmp/iteration-stats-campaign \
  --output /tmp/iteration-stats-campaign/summary/audit.json
MPLCONFIGDIR=/tmp/iteration-stats-matplotlib \
  /tmp/iteration-stats-analysis/.venv/bin/python \
  benchmarks/iteration_stats/plot.py \
  --summary-dir /tmp/iteration-stats-campaign/summary \
  --output /tmp/iteration-stats-campaign/summary/primary.png
```
