# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit a paired serving campaign and retain compact, reproducible results."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from analyze import interval

METRICS = (
    "output_throughput",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_itl_ms",
    "p99_itl_ms",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timezone", default="America/New_York")
    args = parser.parse_args()
    root = args.directory.resolve()
    plan = json.loads((root / "plan.json").read_text())
    zone = ZoneInfo(args.timezone)
    groups = defaultdict(dict)
    runs, sources, harnesses, block_durations = [], defaultdict(set), set(), []
    wall_starts, wall_ends, timing_warnings = [], [], []
    for device in plan["devices"]:
        jobs = json.loads((root / f"gpu{device['gpu']}.json").read_text())
        expected_blocks = plan["repeats"] * len(plan["models"]) * 2
        if len(jobs) != expected_blocks:
            raise ValueError("Incomplete GPU journal")
        for job in jobs:
            directory = root / job["name"]
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest["source_changed"] or manifest["preemptions"]:
                raise ValueError(f"Invalid source or preemption: {directory}")
            if manifest["revision"] != plan[f"{job['label']}_revision"]:
                raise ValueError(f"Unexpected revision: {directory}")
            if manifest["selected_gpu_uuid"] != device["uuid"]:
                raise ValueError(f"Wrong GPU: {directory}")
            if manifest["args"]["port"] != device["port"]:
                raise ValueError(f"Wrong server port: {directory}")
            if (
                manifest["args"]["collection"] != plan["collection"][job["label"]]
                or manifest["args"]["async_mode"] != "on"
            ):
                raise ValueError(f"Unexpected collection/scheduling mode: {directory}")
            sources[job["label"]].add(manifest["source"])
            harnesses.add(manifest["harness_sha256"])
            block_durations.append(manifest["wrapper_wall_seconds"])
            model = next(m for m in plan["models"] if m["name"] == job["model"])
            if len(manifest["runs"]) != len(model["cases"]):
                raise ValueError(f"Incomplete case list: {directory}")
            log = (directory / "server.log").read_text()
            for warning in (
                "without GPU timing",
                "event pool exhausted",
                "Dropping unfinished iteration timings",
                "Could not collect final iteration timings",
            ):
                if warning in log:
                    timing_warnings.append(dict(block=job["name"], warning=warning))
                    if warning == "without GPU timing":
                        raise ValueError(f"Timing disabled: {directory}")
            telemetry = []
            with (directory / "gpu-telemetry.csv").open() as handle:
                for sample in csv.DictReader(handle):
                    sample = {k.strip(): v.strip() for k, v in sample.items()}
                    sample["unix"] = (
                        datetime.strptime(sample["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
                        .replace(tzinfo=zone)
                        .timestamp()
                    )
                    if sample["uuid"] != device["uuid"]:
                        raise ValueError(f"Wrong telemetry GPU: {directory}")
                    telemetry.append(sample)
            for run, expected in zip(manifest["runs"], model["cases"]):
                for key in (
                    "name",
                    "input_len",
                    "output_len",
                    "concurrency",
                    "num_prompts",
                    "num_warmups",
                ):
                    if run[key] != expected[key]:
                        raise ValueError(f"Unexpected {key}: {directory}")
                result = json.loads(
                    (directory / run["name"] / "result.json").read_text()
                )
                if result["completed"] != run["num_prompts"] or result["failed"]:
                    raise ValueError(f"Incomplete requests: {directory}")
                for phase in ("input", "output"):
                    if result[f"total_{phase}_tokens"] != (
                        run["num_prompts"] * run[f"{phase}_len"]
                    ):
                        raise ValueError(f"Unexpected token totals: {directory}")
                start = run["measurement_started_at"]
                duration = result["duration"]
                middle_start, middle_end = (
                    start + 0.1 * duration,
                    start + 0.9 * duration,
                )
                middle = [s for s in telemetry if middle_start < s["unix"] < middle_end]
                if len(middle) < 10:
                    raise ValueError(f"Insufficient GPU telemetry: {directory}")
                row = {k: result[k] for k in METRICS}
                row.update(
                    block=job["name"],
                    gpu=job["gpu"],
                    model=job["model"],
                    case=run["name"],
                    label=job["label"],
                    repetition=job["repetition"],
                    measured_seconds=duration,
                    completed=result["completed"],
                    failed=result["failed"],
                    total_input_tokens=result["total_input_tokens"],
                    total_output_tokens=result["total_output_tokens"],
                    preemptions=run["preemptions"],
                    client_wall_seconds=run["client_wall_seconds"],
                    measurement_started_at=start,
                    measurement_finished_at=run["measurement_finished_at"],
                )
                for field, key in (
                    ("utilization.gpu [%]", "gpu_utilization_percent"),
                    ("power.draw [W]", "gpu_power_watts"),
                    ("clocks.current.sm [MHz]", "gpu_clock_mhz"),
                ):
                    row[key] = statistics.median(
                        float(s[field].split()[0]) for s in middle
                    )
                trace = json.loads(
                    (directory / run["name"] / "client-cpu.json").read_text()
                )
                samples = [
                    s
                    for s in trace
                    if middle_start
                    < run["started_at"] + s["elapsed_seconds"]
                    < middle_end
                ]
                if len(samples) < 2:
                    raise ValueError(f"Insufficient client CPU samples: {directory}")
                row["client_cpu_cores"] = (
                    samples[-1]["cpu_seconds"] - samples[0]["cpu_seconds"]
                ) / (samples[-1]["elapsed_seconds"] - samples[0]["elapsed_seconds"])
                runs.append(row)
                key = (job["gpu"], job["model"], run["name"])
                groups[key].setdefault(job["repetition"], {})[job["label"]] = row
            wall_ends.append(max(r["finished_at"] for r in manifest["runs"]))
            wall_starts.append(wall_ends[-1] - manifest["total_wall_seconds"])
    if len(harnesses) != 1 or any(len(values) != 1 for values in sources.values()):
        raise ValueError("Source or harness changed across runs")
    summaries = []
    for (gpu, model, case), repetitions in sorted(groups.items()):
        pairs = list(repetitions.values())
        if len(pairs) != plan["repeats"] or any(len(p) != 2 for p in pairs):
            raise ValueError("Incomplete paired comparison")
        for metric in METRICS:
            ratios = [
                math.log(p["candidate"][metric] / p["baseline"][metric]) for p in pairs
            ]
            point, ci = interval(ratios)
            summaries.append(
                dict(
                    gpu=gpu,
                    model=model,
                    case=case,
                    metric=metric,
                    pairs=len(pairs),
                    change_percent=point,
                    ci95_percent=ci,
                    paired_log_ratios=ratios,
                    baseline_geomean=statistics.geometric_mean(
                        p["baseline"][metric] for p in pairs
                    ),
                    candidate_geomean=statistics.geometric_mean(
                        p["candidate"][metric] for p in pairs
                    ),
                )
            )
    report = dict(
        plan=plan,
        raw_evidence=str(root),
        method=(
            "Paired Student-t 95% intervals on per-run log ratios; "
            "each GPU analyzed separately"
        ),
        sources={k: next(iter(v)) for k, v in sources.items()},
        harness_sha256=next(iter(harnesses)),
        blocks=len(block_durations),
        cases=len(runs),
        timing_warnings=timing_warnings,
        approximate_campaign_wall_seconds=max(wall_ends) - min(wall_starts),
        wrapper_seconds_range=[min(block_durations), max(block_durations)],
        measured_seconds_range=[
            min(r["measured_seconds"] for r in runs),
            max(r["measured_seconds"] for r in runs),
        ],
        summaries=summaries,
        runs=runs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audited {report['blocks']} blocks, {report['cases']} cases: {args.output}")


if __name__ == "__main__":
    main()
