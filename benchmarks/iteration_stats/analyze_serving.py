# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Add paired confidence intervals to a bounded serving campaign summary."""

import argparse
import json
from pathlib import Path

from analyze import interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    path = args.directory / "summary.json"
    rows = json.loads(path.read_text())
    for row in rows:
        point, bounds = interval(row["paired_log_ratios"])
        row["throughput_change_percent"] = point
        row["throughput_ci95_percent"] = bounds
        print(
            f"GPU {row['gpu']} {row['case']}: throughput {point:+.3f}%, "
            f"95% CI {bounds}, n={row['pairs']}"
        )
    (args.directory / "confidence-intervals.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
