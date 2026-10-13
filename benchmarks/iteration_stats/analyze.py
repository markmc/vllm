# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize paired runs using repetitions, rather than batches, as samples."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import t


def interval(log_ratios):
    mean = statistics.mean(log_ratios)
    point = 100 * math.expm1(mean)
    if len(log_ratios) < 2:
        return point, None
    error = t.ppf(0.975, len(log_ratios) - 1) * statistics.stdev(log_ratios)
    error /= math.sqrt(len(log_ratios))
    return point, [100 * math.expm1(mean - error), 100 * math.expm1(mean + error)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    groups = defaultdict(dict)
    for directory in args.directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        settings = manifest["arguments"]
        if settings["verify_delivery"]:
            raise ValueError("Untimed delivery checks cannot be analyzed as benchmarks")
        for run in manifest["runs"]:
            if args.allow_partial and "returncode" not in run:
                continue
            if run.get("returncode") != 0 or run.get("source_changed"):
                raise ValueError(f"Failed or invalid run: {run['result']}")
            raw = json.loads(Path(run["result"]).read_text())
            key = (
                settings["gpu"],
                settings["cpu_list"],
                settings["model"],
                settings["revision"],
                settings["input_len"],
                settings["output_len"],
                run["batch"],
                run["async"],
            )
            pair_key = (str(directory.resolve()), run["repetition"])
            groups[key].setdefault(pair_key, {})[run["label"]] = (raw, run["source"])
    summaries = []
    for key, repetitions in groups.items():
        for baseline, candidate in (
            ("A", "B"),
            ("B", "C"),
            ("A", "C"),
            ("B", "B-repeat"),
            ("A", "A-enabled"),
        ):
            pairs = [
                values
                for values in repetitions.values()
                if baseline in values and candidate in values
            ]
            if not pairs:
                continue
            identities = {(p[baseline][1], p[candidate][1]) for p in pairs}
            if len(identities) != 1:
                raise ValueError("Cannot pool different implementation snapshots")
            ratios = [
                math.log(p[candidate][0]["avg_latency"] / p[baseline][0]["avg_latency"])
                for p in pairs
            ]
            latency, latency_ci = interval(ratios)
            throughput, throughput_ci = interval([-ratio for ratio in ratios])
            durations = [
                p[label][0]["measured_seconds"]
                for p in pairs
                for label in (baseline, candidate)
            ]
            ready = len(pairs) >= 5 and min(durations) >= 60
            summary = {
                "gpu": key[0],
                "cpu_list": key[1],
                "model": key[2],
                "revision": key[3],
                "input_len": key[4],
                "output_len": key[5],
                "batch": key[6],
                "async": key[7],
                "comparison": f"{candidate}/{baseline}",
                "paired_repetitions": len(pairs),
                "latency_change_percent": latency,
                "latency_ci95_percent": latency_ci,
                "throughput_change_percent": throughput,
                "throughput_ci95_percent": throughput_ci,
                "measured_seconds_range": [min(durations), max(durations)],
                "measurement_stage": "repeated" if ready else "pilot",
                "rules_out_1pct_regression": (
                    ready and latency_ci[1] < 1 and throughput_ci[0] > -1
                ),
                "sources": list(identities.pop()),
            }
            if candidate == "B-repeat":
                summary["noise_abs_changes_percent"] = [
                    abs(100 * math.expm1(ratio)) for ratio in ratios
                ]
            summaries.append(summary)
            print(
                f"GPU {key[0]} batch {key[6]} async {key[7]} {candidate}/{baseline}: "
                f"latency {latency:+.3f}%, 95% CI {latency_ci}, "
                f"n={len(pairs)} ({summary['measurement_stage']})"
            )
    args.output.write_text(
        json.dumps(
            {
                "method": "Student-t interval on paired per-run log latency ratios; "
                "throughput is the reciprocal for fixed output lengths. "
                "Intervals assume independent paired repetitions. "
                "Review warmup stability, noise and serving tails separately.",
                "comparisons": summaries,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
