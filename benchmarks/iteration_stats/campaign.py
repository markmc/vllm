# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Queue control, churn, and serving confirmation after a sensitive sweep."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from run import HERE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-repo", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--server-cpus", required=True)
    parser.add_argument("--client-cpus", required=True)
    parser.add_argument("--confirm-sensitive", action="store_true")
    parser.add_argument("--serving-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    journal = root / f"campaign-gpu{args.gpu}.json"
    if args.confirm_sensitive:
        while True:
            try:
                events = json.loads(journal.read_text())
            except json.JSONDecodeError:
                time.sleep(10)
                continue
            if any(event.get("returncode", 0) for event in events):
                raise RuntimeError("Campaign failed; confirmation stopped")
            if events[-1]["stage"] == "complete":
                break
            time.sleep(10)
        command = [
            sys.executable,
            str(HERE / "matrix.py"),
            "--repo",
            str(args.repo),
            "--base-repo",
            str(args.base_repo),
            "--gpu",
            args.gpu,
            "--cpu-list",
            args.server_cpus,
            "--revision",
            "c1899de289a04d12100db370d81485cdf75e47ca",
            "--cases-file",
            str(root / "sensitive-confirm.json"),
            "--output-dir",
            str(root / f"sensitive-confirm-gpu{args.gpu}"),
            "--configs",
            "A",
            "C",
            "--async-modes",
            "off",
            "--repeats",
            "5",
            "--warmup",
            "8",
            "--seconds",
            "75",
        ]
        print(f"GPU {args.gpu}: targeted sensitive confirmation", flush=True)
        subprocess.run(command, check=True)
        return
    events = []

    def save(event):
        events.append(event)
        journal.write_text(json.dumps(events, indent=2) + "\n")

    sweep = root / f"sensitive-gpu{args.gpu}"
    save(
        {"stage": "starting_serving" if args.serving_only else "waiting_for_sensitive"}
    )
    while not args.serving_only:
        manifests = list(sweep.glob("*/manifest.json"))
        try:
            runs = [
                run
                for path in manifests
                for run in json.loads(path.read_text())["runs"]
            ]
        except json.JSONDecodeError:
            time.sleep(10)
            continue
        if any(run.get("returncode", 0) or run.get("source_changed") for run in runs):
            raise RuntimeError("Sensitive sweep failed; campaign stopped")
        if (
            len(manifests) == 4
            and len(runs) == 80
            and all(run.get("returncode") == 0 for run in runs)
        ):
            break
        time.sleep(10)

    def execute(stage, command):
        save({"stage": stage, "command": command, "status": "running"})
        print(f"GPU {args.gpu}: {stage}", flush=True)
        with (root / f"campaign-gpu{args.gpu}-{stage}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        save({"stage": stage, "returncode": result.returncode, "status": "finished"})
        if result.returncode:
            raise RuntimeError(f"Campaign stage {stage} failed")

    common = [
        sys.executable,
        str(HERE / "matrix.py"),
        "--repo",
        str(args.repo),
        "--base-repo",
        str(args.base_repo),
        "--gpu",
        args.gpu,
        "--cpu-list",
        args.server_cpus,
        "--repeats",
        "5",
        "--warmup",
        "5",
        "--seconds",
        "75",
        "--configs",
        "B",
        "C",
    ]
    if not args.serving_only:
        execute(
            "control",
            common
            + [
                "--model",
                "Qwen/Qwen3-8B",
                "--revision",
                "b968826d9c46dd6066d109eabc6255188de91218",
                "--cases-file",
                str(root / "control.json"),
                "--output-dir",
                str(root / f"control-gpu{args.gpu}"),
                "--async-modes",
                "on",
            ],
        )
        execute(
            "churn",
            common
            + [
                "--revision",
                "c1899de289a04d12100db370d81485cdf75e47ca",
                "--cases-file",
                str(root / "churn.json"),
                "--output-dir",
                str(root / f"churn-gpu{args.gpu}"),
                "--async-modes",
                "on",
                "off",
            ],
        )
    serving = [
        sys.executable,
        str(HERE / "serve.py"),
        "--repo",
        str(args.repo),
        "--gpu",
        args.gpu,
        "--server-cpus",
        args.server_cpus,
        "--client-cpus",
        args.client_cpus,
        "--revision",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "--concurrency",
        "512",
        "--port",
        str(18080 + int(args.gpu)),
    ]
    for mode in ("on", "off"):
        pilot = root / f"serving-pilot-gpu{args.gpu}-async{mode}"
        execute(
            f"serving-pilot-{mode}",
            serving
            + [
                "--collection",
                "off",
                "--async-mode",
                mode,
                "--num-prompts",
                "2048",
                "--output-dir",
                str(pilot),
            ],
        )
        raw = json.loads((pilot / "result.json").read_text())
        count = max(2048, round(75 * 2048 / raw["duration"] / 512) * 512)
        for repetition in range(5):
            order = ("off", "on") if repetition % 2 == 0 else ("on", "off")
            for collection in order:
                output = root / (
                    f"serving-gpu{args.gpu}-async{mode}-r{repetition}-{collection}"
                )
                execute(
                    f"serving-{mode}-r{repetition}-{collection}",
                    serving
                    + [
                        "--collection",
                        collection,
                        "--async-mode",
                        mode,
                        "--num-prompts",
                        str(count),
                        "--output-dir",
                        str(output),
                    ],
                )
    save({"stage": "complete"})
    print(f"GPU {args.gpu}: campaign complete", flush=True)


if __name__ == "__main__":
    main()
