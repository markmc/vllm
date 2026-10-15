# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the existing latency benchmark with an optional untimed delivery check."""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path


def main():
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--repo", type=Path, required=True)
    wrapper.add_argument("--verify-delivery", action="store_true")
    wrapper.add_argument("--require-gpu-timing", action="store_true")
    wrapper.add_argument("--cases-file", type=Path)
    wrapper.add_argument("--target-seconds", type=float)
    options, remaining = wrapper.parse_known_args()
    if options.require_gpu_timing and not options.verify_delivery:
        wrapper.error("--require-gpu-timing needs --verify-delivery")
    repo = options.repo.resolve()
    sys.path.insert(0, str(repo))

    import numpy as np
    import torch

    import vllm
    from vllm import LLM
    from vllm.benchmarks import latency
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.metrics.loggers import StatLoggerManager

    if Path(vllm.__file__).resolve().parent != repo / "vllm":
        raise RuntimeError(f"Wrong vLLM source imported: {vllm.__file__}")
    if os.environ.get("VLLM_LOGGING_LEVEL") != "WARNING":
        raise RuntimeError("Set VLLM_LOGGING_LEVEL=WARNING for all configurations")

    parser = FlexibleArgumentParser()
    latency.add_cli_args(parser)
    args = parser.parse_args(remaining)
    base_args = argparse.Namespace(**vars(args))
    if args.disable_log_stats or not args.disable_detokenize:
        raise ValueError("Enable ordinary log_stats and disable detokenization")
    if args.enable_prefix_caching or args.enforce_eager or args.profile:
        raise ValueError("Disable prefix caching and profiling; keep graphs enabled")
    if not args.output_json or args.num_iters < 1 or args.num_iters_warmup < 1:
        raise ValueError("Require output JSON and positive warmup/measurement counts")

    records = []
    original_record = StatLoggerManager.record
    if options.verify_delivery:

        def record(self, scheduler_stats, *record_args, **record_kwargs):
            if scheduler_stats is not None:
                details = getattr(scheduler_stats, "iteration_records", ())
                if not details and scheduler_stats.iteration_details is not None:
                    details = [scheduler_stats.iteration_details]
                records.extend(dataclasses.asdict(detail) for detail in details)
            return original_record(self, scheduler_stats, *record_args, **record_kwargs)

        StatLoggerManager.record = record

    engines = []
    original_create = LLM.from_engine_args

    def create(cls, engine_args):
        llm = engines[0] if engines else original_create(engine_args)
        assert llm.llm_engine.log_stats
        config = llm.llm_engine.vllm_config
        assert not config.cache_config.enable_prefix_caching
        assert config.profiler_config.profiler is None
        assert config.scheduler_config.async_scheduling == args.async_scheduling
        if not engines:
            engines.append(llm)
        if target_seconds:
            if args.use_beam_search:
                raise ValueError("Duration calibration requires ordinary generation")
            original_generate = llm.generate
            warm_latencies = []

            def warm_generate(*generate_args, **generate_kwargs):
                start = time.perf_counter()
                outputs = original_generate(*generate_args, **generate_kwargs)
                warm_latencies.append(time.perf_counter() - start)
                if len(warm_latencies) == args.num_iters_warmup:
                    llm.generate = original_generate
                    estimate = statistics.mean(warm_latencies[-5:])
                    args.num_iters = max(1, math.ceil(target_seconds / estimate))
                return outputs

            llm.generate = warm_generate
        # LLM initialization can consume NumPy RNG state. Seed the stock
        # benchmark's pretokenized prompt generation after initialization.
        np.random.seed(args.seed)
        return llm

    LLM.from_engine_args = classmethod(create)
    cases = json.loads(options.cases_file.read_text()) if options.cases_file else [{}]
    for case in cases:
        args = argparse.Namespace(**vars(base_args))
        for name in (
            "input_len",
            "output_len",
            "batch_size",
            "num_iters",
            "output_json",
        ):
            if name in case:
                setattr(args, name, case[name])
        target_seconds = case.get("target_seconds", options.target_seconds)
        records.clear()
        latency.main(args)
        llm = engines[0]
        config = llm.llm_engine.vllm_config
        preemptions = sum(
            metric.value
            for metric in llm.llm_engine.get_metrics()
            if metric.name == "vllm:num_preemptions"
        )
        result_path = Path(args.output_json)
        result = json.loads(result_path.read_text())
        result["output_token_throughput"] = (
            args.batch_size * args.output_len / result["avg_latency"]
        )
        result["measured_seconds"] = sum(result["latencies"])
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        metadata = {
            "python": sys.executable,
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "vllm_source": vllm.__file__,
            "args": vars(args),
            "log_stats": llm.llm_engine.log_stats,
            "preemptions": preemptions,
            "async_scheduling": config.scheduler_config.async_scheduling,
            "model_revision": config.model_config.revision,
            "tokenizer_revision": config.model_config.tokenizer_revision,
            "compilation_mode": str(config.compilation_config.mode),
            "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
            "capture_sizes": config.compilation_config.cudagraph_capture_sizes,
            "verify_delivery": options.verify_delivery,
            "require_gpu_timing": options.require_gpu_timing,
            "delivered_records": records if options.verify_delivery else None,
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        result_path.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n"
        )
        if preemptions:
            raise RuntimeError(f"Run invalid: {preemptions} preemptions")
        if options.verify_delivery:
            enabled = config.observability_config.enable_logging_iteration_details
            if bool(records) != enabled:
                raise RuntimeError(
                    f"Delivery check failed: {enabled=}, {len(records)=}"
                )
            indices = [record["iteration_index"] for record in records]
            if len(set(indices)) != len(indices):
                raise RuntimeError(
                    "Duplicate iteration records delivered to the frontend"
                )
            if options.require_gpu_timing and (
                not records
                or any(
                    record.get("timing_scope") != "model_step_cuda"
                    or record.get("elapsed_ms") is None
                    or not math.isfinite(record["elapsed_ms"])
                    or record["elapsed_ms"] < 0
                    for record in records
                )
            ):
                raise RuntimeError("Expected completed GPU timings at the frontend")
            if enabled:
                first = records[0]
                prefill_tokens = (
                    "num_prefill_tokens"
                    if "num_prefill_tokens" in first
                    else "num_ctx_tokens"
                )
                decode_tokens = (
                    "num_decode_tokens"
                    if "num_decode_tokens" in first
                    else "num_generation_tokens"
                )
                if not any(record[prefill_tokens] > 0 for record in records):
                    raise RuntimeError("No prefill counts reached the frontend")
                if args.output_len > 1 and not any(
                    record[decode_tokens] > 0 for record in records
                ):
                    raise RuntimeError("No decode counts reached the frontend")
            print(f"Untimed delivery check: {enabled=}, records={len(records)}")


if __name__ == "__main__":
    main()
