# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the actual scheduler path at a fixed decode context, without delivery."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128, 512])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.v1.core.utils import create_requests, create_scheduler
    from vllm.v1.outputs import ModelRunnerOutput

    results = []
    for batch in args.batch_sizes:
        scheduler = create_scheduler(
            model="Qwen/Qwen3-0.6B",
            max_num_seqs=512,
            max_num_batched_tokens=16384,
            max_model_len=2048,
            skip_tokenizer_init=True,
        )
        requests = create_requests(batch, num_tokens=32, max_tokens=100000)
        for request in requests:
            scheduler.add_request(request)
        prefill = scheduler.schedule()
        scheduler.update_from_output(
            prefill,
            ModelRunnerOutput(
                req_ids=[r.request_id for r in requests],
                req_id_to_index={r.request_id: i for i, r in enumerate(requests)},
                sampled_token_ids=[[0] for _ in requests],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )

        def measure(
            enabled, samples, scheduler=scheduler, requests=requests, batch=batch
        ):
            scheduler.observability_config.enable_logging_iteration_details = enabled
            times = []
            for _ in range(samples):
                # Hold context and KV allocation steady. This reset, and all
                # request/model setup, is outside the schedule() measurement.
                for request in requests:
                    request.num_computed_tokens = 32
                start = time.perf_counter_ns()
                output = scheduler.schedule()
                times.append(time.perf_counter_ns() - start)
            if output.total_num_scheduled_tokens != batch or output.preempted_req_ids:
                raise RuntimeError("Invalid steady-state scheduler workload")
            if enabled:
                assert output.iteration_details.num_decode_requests == batch
                assert output.iteration_details.num_decode_tokens == batch
            return statistics.mean(times)

        measure(False, 100)
        measure(True, 100)
        pairs = []
        for repetition in range(args.repeats):
            order = [False, True] if repetition % 2 == 0 else [True, False]
            values = {enabled: measure(enabled, args.samples) for enabled in order}
            pairs.append(
                {
                    "disabled_ns": values[False],
                    "enabled_ns": values[True],
                    "delta_ns": values[True] - values[False],
                }
            )
        results.append({"batch": batch, "pairs": pairs})
        print(
            f"Batch {batch}: incremental schedule cost "
            f"{statistics.mean(p['delta_ns'] for p in pairs) / 1000:.3f} us",
            flush=True,
        )
    args.output.write_text(
        json.dumps(
            {
                "contract": "Synchronous Scheduler.schedule() with ordinary stats, "
                "fixed 32-token prompt plus one generated token. No model, "
                "serialization, IPC or frontend delivery is measured.",
                "arguments": vars(args),
                "results": results,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
