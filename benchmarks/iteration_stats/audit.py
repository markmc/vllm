# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate completed campaign evidence and report within-run latency drift."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    rows = []
    artifact_sets = []
    harnesses = set()
    sources = {"base": set(), "patch": set()}
    for path in sorted(args.root.glob("*-gpu*/*/manifest.json")):
        if not path.parent.parent.name.startswith(("sensitive-", "control-", "churn-")):
            continue
        manifest = json.loads(path.read_text())
        for label, identity in manifest["sources"].items():
            sources[label].add(identity)
        artifact_sets.extend(manifest["native_artifacts"].values())
        for run in manifest["runs"]:
            if "returncode" not in run and args.partial:
                continue
            assert run.get("returncode") == 0 and not run["source_changed"], path
            result_path = Path(run["result"])
            raw = json.loads(result_path.read_text())
            metadata = json.loads(result_path.with_suffix(".metadata.json").read_text())
            assert metadata["log_stats"] and not metadata["preemptions"], result_path
            assert not metadata["verify_delivery"], result_path
            assert metadata["compilation_mode"] != "NONE", result_path
            assert metadata["capture_sizes"], result_path
            assert metadata["args"]["disable_detokenize"], result_path
            assert not metadata["args"]["enable_prefix_caching"], result_path
            assert metadata["args"]["num_iters"] == len(raw["latencies"]), result_path
            assert abs(statistics.mean(raw["latencies"]) - raw["avg_latency"]) < 1e-9
            harnesses.add(metadata["harness_sha256"])
            count = max(1, len(raw["latencies"]) // 3)
            drift = (
                statistics.mean(raw["latencies"][-count:])
                / statistics.mean(raw["latencies"][:count])
                - 1
            ) * 100
            rows.append(
                {
                    "result": str(result_path),
                    "gpu": manifest["arguments"]["gpu"],
                    "batch": run["batch"],
                    "async": run["async"],
                    "label": run["label"],
                    "samples": len(raw["latencies"]),
                    "measured_seconds": raw["measured_seconds"],
                    "last_vs_first_third_percent": drift,
                }
            )
    assert len(harnesses) == 1, "Measurement harness changed during campaign"
    assert all(len(s) == 1 for s in sources.values()), "Implementation snapshots differ"
    assert all(a == artifact_sets[0] for a in artifact_sets), "Native artifacts differ"
    summary = {
        "validated_runs": len(rows),
        "harness_sha256": list(harnesses),
        "native_artifacts_identical": True,
        "sources": {label: list(identities) for label, identities in sources.items()},
        "run_duration_range_seconds": [
            min(r["measured_seconds"] for r in rows),
            max(r["measured_seconds"] for r in rows),
        ],
        "within_run_drift": rows,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Validated {len(rows)} offline runs; identical harness and native artifacts")


if __name__ == "__main__":
    main()
