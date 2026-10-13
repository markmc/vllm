# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregate completed matrices and paired serving results for the final report."""

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from analyze import interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    directories = sorted(
        path.parent
        for path in root.glob("*-gpu*/*/manifest.json")
        if path.parent.parent.name.startswith(("sensitive-", "control-", "churn-"))
    )
    command = [
        sys.executable,
        str(Path(__file__).with_name("analyze.py")),
        *map(str, directories),
        "--output",
        str(output / "latency.json"),
    ]
    if args.partial:
        command.append("--allow-partial")
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    serving = defaultdict(dict)
    harnesses = set()
    for directory in root.glob("serving-gpu*-async*-r*-*"):
        result_file = directory / "result.json"
        manifest_file = directory / "manifest.json"
        if not result_file.exists() or not manifest_file.exists():
            if args.partial:
                continue
            raise RuntimeError(f"Incomplete serving run: {directory}")
        manifest = json.loads(manifest_file.read_text())
        if "source_changed" not in manifest:
            if args.partial:
                continue
            raise RuntimeError(f"Serving run is not finished: {directory}")
        if manifest["source_changed"] or manifest["preemptions"]:
            raise RuntimeError(f"Invalid serving run: {directory}")
        result = json.loads(result_file.read_text())
        if result["completed"] != manifest["args"]["num_prompts"] or result["failed"]:
            raise RuntimeError(f"Incomplete serving requests: {directory}")
        for phase in ("input", "output"):
            expected = (
                manifest["args"]["num_prompts"] * manifest["args"][f"{phase}_len"]
            )
            if result[f"total_{phase}_tokens"] != expected:
                raise RuntimeError(f"Unexpected serving {phase} lengths: {directory}")
        harnesses.add(manifest["harness_sha256"])
        settings = manifest["args"]
        repetition = int(directory.name.split("-r")[-1].split("-")[0])
        key = (settings["gpu"], settings["async_mode"], settings["concurrency"])
        serving[key].setdefault(repetition, {})[settings["collection"]] = (
            result,
            manifest,
        )
    if len(harnesses) > 1:
        raise RuntimeError("Serving measurement harness changed during campaign")
    serving_summary = []
    for key, repetitions in serving.items():
        pairs = [
            pair for pair in repetitions.values() if "on" in pair and "off" in pair
        ]
        if not pairs:
            continue
        sources = {p[s][1]["source"] for p in pairs for s in ("on", "off")}
        if len(sources) != 1:
            raise RuntimeError("Cannot pool different serving implementation snapshots")
        for metric in (
            "output_throughput",
            "median_ttft_ms",
            "p99_ttft_ms",
            "median_itl_ms",
            "p99_itl_ms",
        ):
            ratios = [
                math.log(pair["on"][0][metric] / pair["off"][0][metric])
                for pair in pairs
            ]
            point, ci = interval(ratios)
            serving_summary.append(
                {
                    "gpu": key[0],
                    "async": key[1],
                    "concurrency": key[2],
                    "metric": metric,
                    "paired_repetitions": len(pairs),
                    "change_percent": point,
                    "ci95_percent": ci,
                    "duration_range": [
                        min(p[s][0]["duration"] for p in pairs for s in ("on", "off")),
                        max(p[s][0]["duration"] for p in pairs for s in ("on", "off")),
                    ],
                }
            )
    (output / "serving.json").write_text(json.dumps(serving_summary, indent=2) + "\n")
    print(f"Summaries saved to {output}")


if __name__ == "__main__":
    main()
