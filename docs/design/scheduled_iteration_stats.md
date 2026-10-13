# Scheduled Iteration Statistics

Status: current-phase design decisions recorded. Implement request/token counts
first and worker execution timing next. Implementation benchmarks determine
default enablement. Context-length, queue, and orchestrator reporting extensions
are follow-on workstreams.

Public RFC: [#61423](https://github.com/vllm-project/vllm/issues/61423).

## Problem statement

Define iteration metrics as statistics relating to the execution of **the
target model's scheduled batch**. Capture them efficiently and use the same
facts for console logging and Prometheus observations, establishing a foundation
for future orchestrator integrations.

The existing console iteration details describe scheduled batch composition,
while some Prometheus observations use frontend request-output accounting.
These answer different questions. In particular, scheduled input tokens can
differ from returned output tokens and from prompt-token accounting reported
when prefill completes. Sharing field names without sharing their definitions
would preserve this ambiguity.

This phase covers the existing iteration statistics and their collection,
semantics, timing, delivery, and frontend consumption. The core quantities are
scheduled prefill/decode request counts, scheduled input-token counts, and the
batch execution interval. It also clarifies the separate role of frontend
request-output accounting currently named `IterationStats`.

The longer-term objective is to support orchestrators such as NVIDIA Dynamo,
as expressed in
[RFC #38760](https://github.com/vllm-project/vllm/issues/38760) and
[PR #52061](https://github.com/vllm-project/vllm/pull/52061), using the same
iteration facts. Low-latency reporting, new per-iteration context-length
statistics, and new scheduler queue statistics build on this phase; their
definitions and export mechanisms belong to follow-on workstreams.

## Objectives

- Establish one clear meaning of an iteration and precise definitions for its
  existing request/token counts and execution interval.
- Reduce the overhead of collecting iteration details so default collection
  with `log_stats` can be considered on the basis of benchmark evidence.
- Unify console and Prometheus observations around shared scheduled-batch facts,
  while preserving request-level prompt, output-token, and lifecycle accounting.
- Clarify frontend data-model names and preserve one observation owner for each
  engine's iteration record, including asynchronously delivered durations.
- Provide a foundation for orchestrator enhancements without requiring new
  context/queue metrics or an external reporting protocol in this phase.

## Current scope

1. Define iteration metrics around a logical target-model scheduled batch,
   including execution identity and explicit non-request activity classification.
2. Reduce the collection overhead associated with
   `enable_logging_iteration_details`. Collect shared facts once and resolve
   collection needs independently of emitting verbose console lines.
3. Add Prometheus histograms for the existing iteration statistics, beginning
   with scheduled prefill/decode request and input-token counts, followed by
   execution duration. Enable them with `log_stats` by default only when
   implementation benchmarks show acceptable overhead.
4. Correct the existing `vllm:iteration_tokens_total` histogram to observe the
   scheduled inputs of each eligible batch, following and crediting #51261.
5. Replace the CPU result-wait measurement in `elapsed_ms` with the worker's
   batch execution interval, including asynchronous duration delivery. Sequence
   this work after the request/token counts.
6. Clarify the semantics of frontend `IterationStats` and rename the struct or
   its fields where necessary to express request-output accounting accurately.
7. Deliver each engine-global iteration record to one frontend for observation,
   preserving and extending the existing `SchedulerStats`/iteration-details
   routing rather than broadcasting the record to all frontends.
8. Preserve batch identity and composition so future consumers can correlate
   them with timing and extend the record with additional scheduler facts.

Collection costs must remain explicit and bounded. Frontend aggregation and
export should follow the existing
[metrics collection architecture](metrics.md#metrics-collection).

## Decision: use iteration metrics for the shared contract

Prefer **iteration metrics** to **forward-pass metrics** for this shared
collection. One scheduled batch can involve multiple model invocations through
microbatching and speculative drafting. Its execution interval also includes
sampling and CPU-submission gaps inside the timing boundaries. The scheduled
batch is the unit that connects these activities and their metrics.

This is a terminology choice in favor of the scheduled-batch contract. FPM in
[PR #52061](https://github.com/vllm-project/vllm/pull/52061) remains a source of
useful measurement and integration work; renaming its external protocol is
outside this phase. Individual target or draft model forward calls do not each
create another iteration record.

## Decision: clarify frontend statistics and observation ownership

The frontend's existing `IterationStats` accumulates request-output and
lifecycle observations while processing one received `EngineCoreOutputs`
packet. It includes returned tokens, prompt accounting, TTFT/ITL observations,
and finished-request statistics. It does not define the target model's scheduled
batch and can be absent when a real scheduled batch produces no request outputs.

Retain that request-output accounting role and clarify its names and contract.
A name such as `RequestOutputStats` is an implementation option. Reserve
iteration terminology in the shared metrics contract for scheduled-batch facts,
and keep transport-packet boundaries distinct from execution identity. New batch
histograms must consume the shared iteration record rather than reconstructing
it from the frontend accumulator.

Engine-global scheduler and iteration details already go to one frontend per
engine update. Preserve that ownership for new iteration records, including
records completed by later worker timings and stats-only deliveries. Each record
contributes at most one observation per instrument. Frontend request-output
accumulators remain local to the frontends handling those requests; this routing
rule applies to the engine-global iteration records.

See [frontend request-output accounting](../../vllm/v1/metrics/stats.py),
[frontend output handling](../../vllm/v1/engine/async_llm.py), and
[scheduler stats routing](../../vllm/v1/core/sched/scheduler.py).

## Decision: use prefill/decode terminology

Use **prefill requests** and **decode requests**, rather than **context requests**
and **generation requests**, to describe the scheduled batch breakdown.

The breakdown reflects **the scheduler's view of each request's execution phase
in this iteration**. A request's category is an iteration-level property, rather
than a permanent request type. A long prompt can contribute a prefill request
to each iteration in which a chunk is scheduled.

This vocabulary aligns with vLLM's
[chunked-prefill documentation](../configuration/optimization.md#chunked-prefill)
and [disaggregated-prefill terminology](../features/disagg_prefill.md).
Context/generation are also accepted phase names; TensorRT-LLM explicitly
[uses them as aliases for prefill/decode](https://nvidia.github.io/TensorRT-LLM/features/disagg-serving.html).
The choice here improves consistency and reduces ambiguity:

- "Context tokens" can suggest the existing attention-context length, whereas
  these iteration token counts describe scheduled inputs.
- "Generation tokens" can suggest newly generated output, whereas scheduled
  decode inputs can include speculative tokens that are later rejected.

The terminology does not identify an attention kernel or classify work solely
by query length. A single scheduled input can belong to prefill, and a decode
request can have multiple scheduled inputs.

### Relationship to P/D worker roles

A worker's configured prefill/decode role does not determine these categories.
The statistics describe the scheduler's phase classification on that engine.
Consequently, the contract does not promise `num_prefill_requests == 0` on a
decode worker or `num_decode_requests == 0` on a prefill worker.

For example, a decode worker can execute residual prompt processing or
recompute prompt KV after a transfer failure. Producing the first output token
in a final prefill execution does not itself make that execution decode.

### Relationship to existing implementations

The existing iteration logger and #52061 use the same scheduled-request
partition: worker-new requests and continuing requests with zero recorded
output tokens count as context/prefill; other continuing requests count as
generation/decode. The recorded output count includes asynchronous placeholders.

See [compute_iteration_details](../../vllm/v1/utils.py),
[CachedRequestData.is_context_phase](../../vllm/v1/core/sched/output.py), and
[#52061's scheduled aggregation](../../vllm/v1/metrics/forward_pass_metrics.py).
The [original iteration-logging PR](https://github.com/vllm-project/vllm/pull/31193)
also describes a context request as being in prefill.

Adopting prefill/decode terminology alone does not change this partition.
The phase decisions below specify the intended contract, including corrections
to the existing recomputation accounting.

## Decision: remote-prefill admission

**A newly admitted request on D counts as a prefill request on its first
scheduled execution after remote KV transfer, including any rewind.**

This matches both the existing iteration-details logging and the FPM
implementation in #52061.

## Decision: preemption and recomputation

**A request re-enters prefill while rebuilding its context after preemption,
even if it has already generated output. Every recomputation chunk counts as
prefill; the request returns to decode when generation continues.**

The rebuilt context can include both the original prompt and previously
generated tokens. The final recomputation chunk still counts as prefill even
if that execution produces a new output token. This rule applies equally to
MRV1 and MRV2.

### Existing console and FPM inconsistencies

For a request preempted after generating output, both iteration-details console
logging and FPM currently report:

| Scheduled work | Intended classification | MRV1 today | MRV2 today |
| --- | --- | --- | --- |
| First recomputation chunk | Prefill | Decode | Prefill |
| Subsequent recomputation chunks | Prefill | Decode | Decode |
| Generation after recomputation | Decode | Decode | Decode |

Preemption preserves the scheduler's output-token history. MRV1 sends resumed
requests as cached requests, so their nonzero output count makes every
recomputation chunk count as decode. MRV2 sends the first resumed chunk as a
worker-new request, which makes it count as prefill. Subsequent chunks are
cached requests with nonzero output counts and therefore count as decode.

Recomputation can be chunked in both runners. The scheduler tracks unfinished
chunks through `is_prefill_chunk`, but the existing metrics classifiers use
worker-new membership and the zero-output test instead. If preemption occurs
before any output, the continuing chunks have zero recorded outputs and count
as prefill in both runners.

See [preemption, resumption, and chunk tracking](../../vllm/v1/core/sched/scheduler.py).
The console and FPM classifier sources are linked above. Worker-state
initialization versus reuse must not determine the shared phase accounting.

### Relationship to request prompt accounting

The scheduler's `PrefillStats` deliberately accounts for the initial prompt
once and excludes repeat prefills after preemption. It is consumed when first
emitted; MRV2 worker-state reconstruction does not create another scheduler
`PrefillStats` object.

The iteration statistics count scheduled work, including recomputation.
Request-level prompt accounting continues to count the initial prompt once.
Classifying recomputation as iteration prefill does not change that accounting.
See [PrefillStats](../../vllm/v1/metrics/stats.py) and
[Request.take_prefill_stats](../../vllm/v1/request.py).

## Decision: streaming input

**Processing a new input chunk starts another prefill phase, even if the
request has already generated output. The request returns to decode when
generation continues.**

Streaming input supplies additional input to an ongoing request. It can retain
the computed prefix while processing the new inputs; recomputation rebuilds
previously known context. Both allow a request to return to prefill after output.

This matches current streaming-input accounting: the scheduler extends the
prompt, clears the current output-token list, and returns the request to
waiting. Its input-processing chunks consequently pass the prefill check.
See [Scheduler._update_request_as_session](../../vllm/v1/core/sched/scheduler.py).

## Decision: asynchronous scheduling

**Determine the phase at scheduling time. A generation step counts as decode
even if output from the preceding prefill execution has not reached EngineCore.**

The final prefill execution remains prefill. Following generation steps count
as decode without waiting for the first sampled token to return to the CPU.
This applies to initial prefill, streaming-input prefill, and recomputation.
Later output arrival does not retroactively change an iteration's phase.

Current accounting already supports this transition by including pending
output placeholders in the cached request's recorded output count. These
placeholders indicate scheduling progress; they are not returned output tokens.
See [AsyncScheduler](../../vllm/v1/core/sched/async_scheduler.py) and the cached
request snapshot in [Scheduler](../../vllm/v1/core/sched/scheduler.py).

## Decision: iteration scope and observation

**Each iteration record describes the target model's scheduled batch.
Its prefill/decode request and token counts are fixed at scheduling time.**

For scheduled request work, an iteration is the logical target-model execution
scheduled through `_process_engine_step()`. Internal microbatching and additional
model invocations within that execution do not create more iteration records.

With async scheduling, a call can schedule a new iteration while collecting an
older iteration's output. The new execution's future enters `batch_queue` when
scheduled, while its `EngineCoreOutputs` can reach the output queue in a later
call. Request counts remain associated with the batch that was scheduled,
independently of the call that reports its output.

Count each scheduled request in its phase at scheduling time. A later abort or
speculative rejection does not remove it from that batch's request counts.
Reporting results and capturing scheduled batch composition are separate events.

### Iteration identity

Reuse the existing iteration-details `iteration_index`, scoped to an engine's
lifetime and local DP rank. It identifies the execution whose stats are reported,
rather than the `_process_engine_step()` call that delivers them. There is no
need to switch to the scheduler's `current_step`, which also counts empty
scheduling attempts.

Existing console logging also indexes dummy execution. Retain those execution
records and their indices, with an explicit classification as described below.
Consumers that filter them out can observe gaps in the indices; the index
identifies an iteration and does not prescribe the number of histogram
observations. Preserve the identity when associating later timing with the same
execution.

See [EngineCore scheduling, capture, and output delivery](../../vllm/v1/engine/core.py).

## Decision: capture and mark non-request activity

**Capture stats for dummy execution and other non-request activity, and mark
the records explicitly so consumers can select which kinds to observe.**

Distinguish the following kinds of activity:

| Activity | Record meaning |
| --- | --- |
| Scheduled request execution | An iteration with scheduled prefill/decode requests and inputs. |
| Dummy execution | Model execution without scheduled request work, such as DP coordination. |
| Empty scheduling step | A scheduling attempt with no request inputs assigned. |
| Idle heartbeat | An engine-state observation with no scheduled model execution. |

Dummy and other non-request records have zero scheduled prefill/decode request
and token counts. Empty steps and heartbeats do not imply a forward pass.
The record's classification must identify the activity directly; consumers must
not infer it from zero counts or timing values. The exact marker representation
is an implementation choice.

Collection and metric observation are separate decisions. For example,
Prometheus histograms of scheduled request batch sizes can exclude dummy,
empty-step, and heartbeat records while console logging or diagnostic consumers
retain them. This preserves visibility into engine activity without adding
non-request records to request-work distributions.

Existing console iteration details already have an `is_dummy` marker. The FPM
payload in #52061 carries empty scheduled counts for dummy execution and idle
heartbeats but has no explicit activity marker. The shared stats contract must
make this distinction explicit for all consumers.

## Decision: token counts

**Count the input token positions assigned by the scheduler to requests in
each phase, fixed at scheduling time.**

For each request, use its `SchedulerOutput.num_scheduled_tokens` value from the
finalized scheduling decision. Assign all those positions to the request's
phase for that iteration. Consequently:

```text
num_prefill_tokens + num_decode_tokens
    = total_num_scheduled_tokens at scheduling time
```

- Count the scheduled chunk, including recomputation and rewind/replay work.
- Exclude reused prefix-cache or remote-KV positions unless scheduled for replay.
- Include scheduled speculative positions, even if rejected after verification
  or trimmed by the worker before execution.
- Retain the counts if output is discarded after an abort or truncated by a
  stop/length limit.
- Exclude additional padding introduced by the model runner beyond the
  scheduler-assigned positions.

These statistics describe input positions in **the target model's scheduled
batch**. They do not measure returned output tokens, existing KV-context lengths,
or all work performed by encoder and drafter model executions. Scheduled inputs
can differ from the inputs actually executed after worker adjustments.

With a draft model, scheduled speculative positions count its proposed tokens
as inputs for verification by the target model. The draft model's own prefill
and generation executions do not add to these four counts or create separate
iteration records. Separate draft-model workload accounting is not defined by
this design. The execution-duration scope defined below includes drafting for
the next batch, without adding draft-model work to these four counts.

### Relationship to console logging and FPM

This preserves the token quantity used by existing iteration-details console
logging: both context and generation token counts sum scheduler-assigned input
positions. FPM's `sum_prefill_tokens` uses the same quantity. FPM has no
equivalent decode-input count; its `sum_decode_kv_tokens` measures prior context
length instead.

MRV1's GPU n-gram path copies scheduling metadata before trimming invalid draft
slots, preserving EngineCore's original counts. MRV2's adaptive verification
compacts worker-side inputs without changing the original scheduler counts.
See [MRV1](../../vllm/v1/worker/gpu_model_runner.py),
[draft trimming](../../vllm/v1/spec_decode/ngram_proposer_gpu.py), and
[adaptive verification](../../vllm/v1/worker/gpu/spec_decode/adaptive_verification.py).

Post-verification rejection does not reduce the scheduled input counts. FPM's
async rejection correction adjusts context lengths, a separate statistic.
The agreed recomputation rule changes which phase receives scheduled tokens,
while preserving their total.

## Initial statistics

All four counts describe the target model's scheduled batch. The agreed meanings
and working field names are:

| Existing field | Working name | Meaning |
| --- | --- | --- |
| `num_ctx_requests` | `num_prefill_requests` | Number of requests scheduled in the scheduler's prefill phase for this iteration. |
| `num_ctx_tokens` | `num_prefill_tokens` | Number of input token positions scheduled for those requests in this iteration. |
| `num_generation_requests` | `num_decode_requests` | Number of requests scheduled in the scheduler's decode phase for this iteration. |
| `num_generation_tokens` | `num_decode_tokens` | Number of input token positions scheduled for those requests in this iteration. |

Request counts measure participation in this batch. Token counts describe
scheduled inputs, including scheduled speculative inputs, rather than accepted
or returned output tokens. Existing context lengths are separate statistics.

These are per-iteration values; calling them counts does not prescribe a
Prometheus instrument type. The Prometheus mapping is specified below; console
wording, internal field migration, and compatibility details will follow in
implementation.

## Decision: Prometheus observations

**Observe collected request/token statistics as histograms using the shared
iteration record and existing single-frontend stats routing.**

The desired default is collection with `log_stats`, independently of verbose
iteration console logging. That default is conditional on acceptable measured
overhead. New per-iteration collection remains opt-in until benchmarking
justifies enabling it with `log_stats`. The same gate applies separately to
request/token counts and execution timing; timing adds a Prometheus histogram
when enabled. See the collection-enablement decision below.

Resolve collection requirements before starting EngineCore so the shared stats
are available to the frontend loggers.

### Instruments and observation population

Follow the phase-labelled histogram families proposed in
[#53550](https://github.com/vllm-project/vllm/pull/53550):

- `vllm:iteration_requests{phase="prefill"|"decode"}` records the request counts.
- `vllm:iteration_tokens_by_phase{phase="prefill"|"decode"}` records the
  scheduled-input token counts.

Each eligible target-model batch contributes one observation to each of the
four phase/statistic combinations, including zero when a phase is absent.
Include a zero bucket to distinguish phase absence. Both phases therefore
describe the same population of scheduled request iterations.

Filter dummy execution, empty scheduling steps, and idle heartbeats using the
record's activity marker. Their stats remain available to other consumers.
Reuse the existing `SchedulerStats` delivery to one frontend per engine
iteration, so multiple frontends do not duplicate histogram observations.

### Iterations without request outputs

**A scheduled batch contributes observations even when it produces no
per-request outputs.** This is ordinary behavior for batches containing only
intermediate prefill or recomputation chunks.

With `log_stats` enabled, the scheduler already creates a stats-only
`EngineCoreOutputs` packet when there are no request results to send. Its
`outputs` list is empty and its `scheduler_stats` field is populated. The
frontend passes these stats to its loggers even though frontend `IterationStats`
is absent. Observe the shared iteration record from `SchedulerStats`, rather
than requiring request outputs or frontend `IterationStats`.

An async engine-step call that schedules a batch but has no completed batch to
report may return no packet yet. The completed batch's stats are delivered
later. This is distinct from a completed iteration with an empty request-output
list.

No empty-output-iteration counter or inference from iteration-index gaps is
needed to account for these batches. Eligibility follows the explicit iteration
record and its activity classification, not whether `EngineCoreOutputs.outputs`
is empty. The histogram observation counts already count eligible iterations.

See [Scheduler.update_from_output](../../vllm/v1/core/sched/scheduler.py),
[EngineCore iteration attachment](../../vllm/v1/engine/core.py), and
[AsyncLLM output handling](../../vllm/v1/engine/async_llm.py).

### Correct the existing aggregate token histogram

Fix `vllm:iteration_tokens_total` to observe
`num_prefill_tokens + num_decode_tokens` once per eligible iteration, following
and crediting [#51261](https://github.com/vllm-project/vllm/pull/51261).

Its current frontend accounting omits intermediate prefill-only iterations and
combines completed-prefill compute accounting with returned generation tokens.
Keep the existing metric name, type, labels, and buckets, and update its HELP
text and documentation to describe scheduled target-model inputs. Its `_sum`,
`_count`, and distribution will change with the corrected semantics.
Request-level prompt and generated-output counters retain their separate
definitions.

## Decision: implement execution timing after the counts

Implement the shared request/token counts, console integration, and Prometheus
observations first. Add worker-side execution timing in a subsequent change,
with overhead benchmarking before deciding how broadly to enable it.

### Collection enablement and overhead

**Request/token counts and execution timing are off by default unless overhead
benchmarking supports enabling them with `log_stats`.** The preferred outcome
is that both are available with `log_stats`, without requiring iteration console
logging. Benchmark counts first, then worker timing, including their consumer
and delivery costs, so timing overhead does not prevent enabling inexpensive
counts.

Choose the final activation policy from implementation benchmark results; this
is an implementation acceptance gate, rather than an open design decision.
One possible outcome is that counts provide sufficient routine operational
visibility and become available with `log_stats`, while execution timing remains
enabled only by the iteration-logging option. In that case the option enables
both execution-duration console logging and a Prometheus duration histogram.
Collection requirements must remain independent of external FPM publication.

The agreed interpretation of **scheduled batch execution time** is the elapsed
interval on the worker's CUDA stream from immediately before target-model
execution through sampling and speculative drafting for the next batch. CPU
delays that leave the stream idle inside these boundaries contribute to the
duration. Work queued before the start event does not. This is an elapsed
interval, rather than a sum of kernel execution times or independent work on
other streams.

Use a pair of CUDA events and collect completed durations without adding
blocking GPU synchronization for telemetry. Request outputs can become
available before drafting and the end event finish.

### Asynchronous iteration-record delivery

**Send request outputs as soon as they are available. When timing is enabled,
deliver the complete iteration record once its duration is available.**

EngineCore retains the scheduling snapshot and associates the worker's completed
duration with it using the engine-local iteration identity. Include the complete
record in a later `EngineCoreOutputs` when necessary. Only telemetry is delayed;
request-output delivery does not wait for the timing event. When only counts
are collected, their record can accompany the corresponding batch's output
without waiting for a duration.

Allow completed records to travel in a stats-only packet when no later request
outputs are available, and allow a packet to carry multiple ready iteration
records. Reuse the existing single-frontend stats routing. A record retains the
identity, activity kind, and batch composition of its originating iteration,
regardless of which output packet carries it. Do not mistake it for the packet's
current request-output iteration or regenerate its scheduling snapshot later.

Console logging and histograms consume each delivered iteration record once.
In the timing-enabled path, count observations may therefore arrive later too;
do not observe the counts again merely because their duration became available.
Preserve the existing activity filtering for scheduled-request histograms.

### Existing `elapsed_ms`

`SchedulerIterationDetails.elapsed_ms` is an EngineCore CPU interval around
result retrieval, including the separate sampling call on the nonqueued path.
It starts after execution submission. It can approximate batch latency when
only one batch is outstanding and waiting for its result dominates the interval,
but it excludes work before the timer starts and can miss drafting that continues
after outputs become available.

With queued execution, it measures only the remaining result wait when
EngineCore retrieves the batch. Substantial GPU execution can therefore produce
an almost-zero `elapsed_ms`. Async scheduling and pipeline parallelism can both
enable this queue. Do not interpret this field as the agreed scheduled batch
execution time.

**Replace this CPU result-wait measurement with the worker execution duration
in iteration console logging when worker timing is implemented.** There is no
requirement to preserve both timing quantities. The field name is an
implementation choice; retaining `elapsed_ms` does not require retaining its
old measurement. When execution timing is disabled or unavailable, omit the
execution duration rather than substituting the CPU wait or a zero value.

See [EngineCore timing boundaries](../../vllm/v1/engine/core.py),
[batch concurrency](../../vllm/config/vllm.py), and
[FPM's event timer](../../vllm/v1/metrics/forward_pass_metrics.py).

## Related issues and pull requests

These references provide implementation history, overlapping work to coordinate,
and requirements for later extensions. The follow-on proposals do not expand
the current scope.

### Iteration logging and Prometheus accounting

| Reference | Relevance |
| --- | --- |
| [PR #31193: Add iteration level logging and enhance NVTX marker](https://github.com/vllm-project/vllm/pull/31193) | Introduced EngineCore iteration logging, context/generation request and token counts, elapsed time, and engine-local iteration indices. |
| [PR #34860: Add KV cache usage metrics to iteration logging](https://github.com/vllm-project/vllm/pull/34860) | Earlier proposal to enrich iteration logs with KV usage; useful history for the later frontend refactor. |
| [PR #46647: Move iteration logging to the frontend](https://github.com/vllm-project/vllm/pull/46647) | Established `SchedulerIterationDetails` delivery through `SchedulerStats` and frontend logging; also added scheduled encoder input/embedding counts. |
| [PR #53550: Add phase-aware engine iteration histograms](https://github.com/vllm-project/vllm/pull/53550) | Directly relevant request/token histograms and a logger collection-requirements contract independent of verbose iteration logging. |
| [PR #51261: Report iteration_tokens_total for each chunked prefill](https://github.com/vllm-project/vllm/pull/51261) | Corrects the aggregate histogram to observe scheduled inputs for every batch, including intermediate prefill chunks. This design follows and credits that work. |
| [PR #43346: Exclude KV transfer tokens from iteration_tokens_total](https://github.com/vllm-project/vllm/pull/43346) | Earlier correction using locally computed prompt-token accounting. It addresses transferred-token inflation, while the per-batch observation correction is covered by #51261. |
| [PR #39456: Add num_requests_prefilling/decoding Prometheus gauges](https://github.com/vllm-project/vllm/pull/39456) | Related phase-count proposal. Instantaneous running-request gauges describe a different population from the scheduled-batch histograms defined here. |
| [PR #37576: Add Prometheus counter for CUDA graph mode](https://github.com/vllm-project/vllm/pull/37576) | Complementary iteration-level execution-mode visibility and separation of cheap collection from verbose logging. |

### Execution timing, FPM, and benchmarking

| Reference | Relevance |
| --- | --- |
| [RFC #38760: Per-iteration forward pass metrics with accurate engine-level timing](https://github.com/vllm-project/vllm/issues/38760) | Orchestrator requirements for correlated scheduled-work, timing, context, and queue telemetry; motivation for a shared iteration-statistics foundation. |
| [PR #52061: Add native forward-pass metrics emission](https://github.com/vllm-project/vllm/pull/52061) | Worker CUDA-event timing through sampling/drafting, delayed timing association, scheduler snapshots, and external FPM publication. Its measurement work informs this design; an external export protocol is follow-on scope. |
| [RFC #55259: Optional native startup self-benchmarking](https://github.com/vllm-project/vllm/issues/55259) | Startup performance-model calibration from scheduler-shaped prefill/decode points and FPM timing. A future consumer of the shared facts and timing scope. |
| [PR #61339: Add vllm bench forward: exact-shape forward timing](https://github.com/vllm-project/vllm/pull/61339) | Standalone exact-shape, captured model-forward benchmarking. Its narrower model-forward timing scope differs from the serving batch interval defined here. |
| [PR #48292: Log result wait time in iteration details](https://github.com/vllm-project/vllm/pull/48292) | Explicit CPU `future.result()` wait measurement; useful comparison with the worker execution interval replacing `elapsed_ms`. |
| [PR #50919: MRV2 worker-level timings](https://github.com/vllm-project/vllm/pull/50919) | Earlier worker timing proposal covering model-runner and speculative proposer durations, metric export, and iteration logging. Prior work for timing integration and overhead evaluation. |

### Broader observability and queue extensions

| Reference | Relevance |
| --- | --- |
| [RFC #39979: Ultimate Better Observability](https://github.com/vllm-project/vllm/issues/39979) | Broader request/iteration observability using coarse scopes and OpenTelemetry, with an emphasis on online monitoring and collection overhead. |
| [PR #59097: Export the prefill token backlog as num_queued_tokens](https://github.com/vllm-project/vllm/pull/59097) | Frontend admission/backlog accounting based on full prompt lengths before first output. Complementary to future scheduler queue statistics, whose remaining-work semantics still need definition. |

## Follow-on workstreams

The previous context-length, queue-statistics, and orchestrator-reporting open
decisions are closed for this phase by assigning them to the following
workstreams. Their detailed contracts remain to be developed there; the current
phase does not adopt FPM's existing definitions for those additional quantities.

- **Per-iteration context-length statistics:** define the logical context
  associated with scheduled prefill/decode work, its snapshot boundary, and any
  sums or variance needed by consumers. Distinguish context length from scheduled
  input counts, physical KV occupancy, and actual attention/cache reads.
- **Scheduler queue statistics:** define the population, runnable versus blocked
  requests, execution-phase classification, remaining known work, and observation
  boundary. Waiting status, execution phase, and remaining work are different
  concepts; existing running/waiting gauges retain their current roles.
- **Orchestrator reporting:** establish requirements for low-latency delivery,
  freshness, paired observations, sampling, loss, and external identity before
  choosing an export contract or transport. Support Dynamo and other consumers
  through the shared collection and existing frontend/exporter architecture.

These workstreams can build on the iteration identity, shared batch record,
asynchronous duration association, activity classification, and observation
ownership established here. They do not block the current implementation.

There are no remaining open design decisions for the current phase. Default
enablement is determined by implementation overhead benchmarks, as agreed above.

## Implementation sequencing

Build small, independently reviewable changes. Start with the count-collection
foundation and implement execution timing last. Keep unrelated refactoring out
of the collection and metric changes.

### 1. Efficient request/token count collection and semantics

Limit the first PR to classification, efficient collection, and the existing
iteration-details payload. Implement the precise request/token definitions in
this document, including the required recomputation corrections for MRV1 and
MRV2. Preserve iteration identity, activity classification, and single-frontend
observation ownership.

Benchmark collection enabled versus disabled with verbose iteration logging
off, including the cost of delivery. Use that evidence to determine whether
collection can be enabled with `log_stats` by default. Do not include histogram
instruments, the frontend struct rename, or asynchronous GPU timing delivery in
this PR.

### 2. Phase-aware request/token histograms

Build on the count collector to add the prefill/decode request and scheduled
input-token histograms. Coordinate with
[#53550](https://github.com/vllm-project/vllm/pull/53550), which already includes
collection setup as well as histogram instruments; separate the shared
collection foundation from the histogram addition where needed.

Benchmark the additional frontend observation cost before choosing default
enablement. Acceptable collection overhead alone does not establish acceptable
histogram overhead.

### 3. Correct the aggregate iteration-token histogram

Advance and credit [#51261](https://github.com/vllm-project/vllm/pull/51261) as a
focused correctness change to `vllm:iteration_tokens_total`. This change and the
phase-aware histogram addition do not depend on each other.

If phase counts are collected by default, reuse their sum. If they remain
opt-in, the scheduler's existing `total_num_scheduled_tokens` can be carried as
a scalar to correct the aggregate histogram without requiring phase-count
collection. The fix therefore does not have to wait for default enablement of
all four counts. Apply the same scheduled-batch observation and activity rules
in either case.

### Independent frontend naming cleanup

Rename the frontend `IterationStats` in a separate, small PR that can merge
early and does not block count collection. `RequestOutputStats` is a suitable
working name. Clarify its existing request-output accounting responsibility and
update names and consumers consistently. Keep batch histogram accounting
changes in the dedicated metric fixes.

### 4. Worker execution timing and asynchronous iteration delivery

Implement CUDA-event execution timing and asynchronous delivery of completed
iteration records together. Delayed telemetry delivery is required to report
the agreed execution interval without delaying request outputs. Include
replacing the console `elapsed_ms` measurement and adding the execution-duration
histogram, using the timing and delivery contract above.

Build on the shared count collector and single-frontend route. Benchmark this
stage separately, including worker instrumentation, pending-record association,
delivery, and frontend duration observation, before choosing timing enablement.

The sequence establishes reliable scheduled-batch facts first, with separate
overhead evidence for collection, histogram observation, and worker timing.
Develop context/queue metrics and orchestrator reporting in the follow-on
workstreams above.

## Collection overhead benchmark plan

Use the two available A100 80GB GPUs to measure the phase-1 collector and its
existing delivery path. Histogram observation and CUDA-event execution timing
remain separate benchmark stages.

### Workload rationale

The primary hypothesis is that collection overhead will be most visible with a
small model decoding short contexts at a high iteration rate. GPU execution is
fast, leaving less time to overlap additional CPU work. The collector adds both
fixed per-iteration work and per-request counter updates, so test small and
large batches rather than assuming the largest batch is the worst case. This
addresses the likely reviewer concern about Python bookkeeping in the scheduler
hot path. vLLM's
[performance guidance](https://docs.vllm.ai/en/stable/configuration/optimization/#cpu-resources-for-gpu-deployments)
identifies scheduling latency as a direct contributor to GPU utilization.

Start with [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), FP16, TP=1,
with compilation and CUDA graphs enabled.

| Workload | Model | Input/output tokens | Batch size or concurrency | Purpose |
| --- | --- | --- | --- | --- |
| Decode-heavy, primary | Qwen3-0.6B | 32 / 256 | 1, 32, 128, 512 | Fast iterations and increasing per-request collection work. |
| Request churn | Qwen3-0.6B | 32 / 1 | 32, 128 | Frequent admissions and short iterations. |
| Representative serving control | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 1,024 / 128 | 32, 128 | Check an ordinary serving workload. |

Run the primary sweep with async scheduling enabled and disabled. Async
scheduling may hide some scheduler cost through overlap; synchronous scheduling
makes its effect more directly visible. Use the pilot to identify the most
sensitive batch sizes for repeated measurements.

### Comparison configurations

Keep ordinary `log_stats` enabled in every configuration.

| Configuration | Code | Iteration collection |
| --- | --- | --- |
| A | Main at the patch's base revision | Disabled |
| B | Phase-1 patch | Disabled |
| C | Phase-1 patch | Enabled through `enable_logging_iteration_details` |

Compare B against A to detect regressions in the default path: some phase
classification work currently happens even when collection is disabled.
Compare C against B to isolate the incremental cost of enabling collection.
Compare C against A to measure the overall cost of enabling the new collection
by default. An additional run of main with iteration collection enabled can
quantify the improvement over the legacy implementation.

Set `VLLM_LOGGING_LEVEL=WARNING` in every configuration. This suppresses verbose
console logging while preserving enabled collection and delivery. Redirecting
INFO output to `/dev/null` would still measure formatting and writing costs.
Confirm in a separate untimed check that collection and delivery are active in
configuration C.

### Measurement paths and controls

Start with `vllm bench latency`, using fixed batches, pretokenized inputs, and
detokenization disabled. This reduces unrelated frontend work while exercising
scheduling, execution, and iteration-stat delivery. Explicitly keep `log_stats`
enabled when using the offline engine.

Confirm the most sensitive cases with `vllm bench serve`, using fixed random
input/output lengths, `--ignore-eos`, and `--request-rate inf` with bounded
concurrency. The
[serving benchmark](https://docs.vllm.ai/en/latest/cli/bench/serve/)
supports these controls and saving detailed results.

Keep prefix caching disabled, avoid preemption, and leave profilers off. Keep
all other settings, model revisions, workloads, and seeds identical across
comparisons. Warm compilation and every measured graph shape before timing;
exclude startup and warmup from results.

Record batch latency and output-token throughput. For serving, also record
median and p99 inter-token latency and TTFT. Use benchmark wall-clock
measurements, not the existing `elapsed_ms` field, which is especially
misleading under async scheduling.

### GPU allocation and repetition

Run independent TP=1 experiments on each GPU. Pin their processes to disjoint
physical CPU cores, respecting GPU/NUMA placement, and reserve separate cores
for benchmark clients so host contention does not dominate the result.

Alternate configuration order and repeat every comparison on both GPUs.
Compare configurations within each GPU rather than comparing an enabled run on
one GPU against a disabled run on the other. This prevents hardware differences
from appearing as collection overhead.

Start with five paired repetitions of roughly 60-120 seconds per measured
configuration, increasing repetitions if uncertainty remains material. Include
unchanged-versus-unchanged repetitions to establish measurement noise. Save raw
results and the exact revision, command, configuration, and hardware details
needed to reproduce each comparison.

### Review evidence and enablement

Report percentage changes, confidence intervals, and baseline measurement noise
for the sensitive cases and representative control. A provisional target is to
rule out a 1% throughput or latency regression in the sensitive workload;
absence of statistical significance alone does not establish negligible cost.
Check tail latency for consistent regressions as well.

If GPU measurements cannot resolve the collection cost, add a focused CPU
benchmark of the actual scheduler collection path, with setup outside the timed
region, to quantify that cost. Do not infer negligible overhead solely from a
GPU-bound workload or a noisy result. Default collection enablement remains
conditional on the measured evidence.

## Implementation status

The design phase is complete. A review draft of phase 1 captures request/token
counts in the scheduler before progress advances and carries that snapshot
through the existing iteration-details route. Decode counts accumulate in the
existing cached-request metadata pass after preemption finalizes the batch;
prefill counts are derived from batch totals, without another request-list walk.
Encoder input counts accumulate when requests are scheduled, and encoder output
token counts use each request's encoder-budget delta. Preemption subtracts the
removed inputs and embeddings; no final encoder-input traversal is needed.
The initial draft uses `enable_logging_iteration_details` to gate collection and
console logging; separate collection enablement is deferred. Existing worker
profiling also consumes the scheduler snapshot. The scheduler accumulator uses
prefill/decode field names, with temporary context/generation alias properties
for existing consumers. `SchedulerIterationDetails` retains its original field
names, and console wording retains context/generation naming to defer consumer
naming changes. Existing worker profiler annotation strings and local variables
also retain context/generation naming. The legacy
collector and its worker-cache phase inference have been removed; detailed
profiling uses phases recorded in the existing metadata pass instead.

New semantic tests remain deferred for the initial implementation-direction
review. Full collection overhead comparisons remain planned; initial pilot
runs do not yet establish acceptable overhead. Histograms, the frontend rename,
and worker execution timing remain
subsequent chunks. Default enablement still requires benchmark evidence.
