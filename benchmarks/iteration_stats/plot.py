# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot paired confidence intervals from the campaign summaries."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.summary_dir / "latency.json").read_text())["comparisons"]
    primary = [
        row
        for row in data
        if row["input_len"] == 32
        and row["output_len"] == 256
        and row["comparison"] == "C/A"
    ]
    primary.sort(key=lambda row: (row["async"], row["batch"], row["gpu"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained", sharex=True)
    for axis, mode in zip(axes, ("on", "off")):
        rows = [row for row in primary if row["async"] == mode]
        for y, row in enumerate(rows):
            point = row["latency_change_percent"]
            interval = row["latency_ci95_percent"]
            color = "#0072B2" if row["gpu"] == "0" else "#D55E00"
            axis.plot(point, y, "o", color=color)
            if interval:
                axis.hlines(y, *interval, color=color, linewidth=2)
        axis.set_yticks(
            range(len(rows)),
            [
                f"Batch {r['batch']}, GPU {r['gpu']} (n={r['paired_repetitions']})"
                for r in rows
            ],
        )
        axis.axvline(0, color="gray", linewidth=1)
        axis.axvline(1, color="black", linestyle="--")
        axis.grid(axis="x", alpha=0.2)
        axis.invert_yaxis()
        axis.set_xlabel("Mean batch latency change (%)")
        axis.set_title(f"Async scheduling {mode}; dashed line: 1% threshold")
    fig.suptitle(
        "Phase-1 collector enabled / base disabled\n"
        "Qwen3-0.6B, 32 input / 256 output tokens; paired 95% intervals"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)
    serving = json.loads((args.summary_dir / "serving.json").read_text())
    rows = sorted(
        (row for row in serving if row["metric"] == "output_throughput"),
        key=lambda row: (row["async"], row["gpu"]),
    )
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    for y, row in enumerate(rows):
        color = "#0072B2" if row["gpu"] == "0" else "#D55E00"
        axis.plot(row["change_percent"], y, "o", color=color)
        if row["ci95_percent"]:
            axis.hlines(y, *row["ci95_percent"], color=color, linewidth=2)
    axis.set_yticks(
        range(len(rows)),
        [
            f"Async {r['async']}, GPU {r['gpu']} (n={r['paired_repetitions']})"
            for r in rows
        ],
    )
    axis.axvline(0, color="gray", linewidth=1)
    axis.axvline(-1, color="black", linestyle="--")
    axis.grid(axis="x", alpha=0.2)
    axis.invert_yaxis()
    axis.set_xlabel("Output-token throughput change (%); negative is slower")
    axis.set_title(
        "Serving: collection on/off, paired 95% intervals\n"
        "Qwen3-0.6B, concurrency 512; dashed line: -1% threshold"
    )
    fig.savefig(args.output.with_name("serving.png"), dpi=180)
    fig.savefig(args.output.with_name("serving.svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
