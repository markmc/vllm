# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a bounded, paired serving campaign on independent GPUs."""

import argparse
import concurrent.futures
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from run import HERE, source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--candidate-repo", type=Path, default=Path.cwd())
    parser.add_argument("--minutes", type=float, default=120)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.minutes <= 0 or plan["repeats"] < 2:
        raise ValueError("Require a positive budget and at least two pairs")
    if len({d["port"] for d in plan["devices"]}) != len(plan["devices"]):
        raise ValueError("Each GPU worker needs a distinct HTTP port")
    if len({str(d["gpu"]) for d in plan["devices"]}) != len(plan["devices"]):
        raise ValueError("Each worker needs a distinct GPU")
    output = args.output_dir.resolve()
    repositories = {
        "baseline": args.baseline_repo.resolve(),
        "candidate": args.candidate_repo.resolve(),
    }
    states = {name: source_state(repo) for name, repo in repositories.items()}
    identities = {name: state[2] for name, state in states.items()}
    for name, state in states.items():
        if state[0] != plan[f"{name}_revision"] or state[1]:
            raise ValueError(f"{name} must match the plan's frozen revision")
    jobs = []
    for device_index, device in enumerate(plan["devices"]):
        device_jobs = []
        for repetition in range(plan["repeats"]):
            models = list(plan["models"])
            labels = ["baseline", "candidate"]
            if (repetition + device_index) % 2:
                models.reverse()
                labels.reverse()
            for model in models:
                for label in labels:
                    name = f"gpu{device['gpu']}-r{repetition}-{model['name']}-{label}"
                    directory = output / name
                    cases_file = output / f"{model['name']}-cases.json"
                    command = [
                        sys.executable,
                        str(HERE / "serve.py"),
                        "--repo",
                        str(repositories[label]),
                        "--output-dir",
                        str(directory),
                        "--gpu",
                        str(device["gpu"]),
                        "--server-cpus",
                        device["server_cpus"],
                        "--client-cpus",
                        device["client_cpus"],
                        "--port",
                        str(device["port"]),
                        "--model",
                        model["model"],
                        "--revision",
                        model["revision"],
                        "--collection",
                        plan["collection"][label],
                        "--async-mode",
                        "on",
                        "--dtype",
                        model["dtype"],
                        "--gpu-memory-utilization",
                        str(model["gpu_memory_utilization"]),
                        "--max-num-seqs",
                        str(model["max_num_seqs"]),
                        "--max-model-len",
                        str(model.get("max_model_len", 2048)),
                        "--max-num-batched-tokens",
                        str(model.get("max_num_batched_tokens", 16384)),
                        "--cases-file",
                        str(cases_file),
                    ]
                    if model.get("language_model_only"):
                        command.append("--language-model-only")
                    device_jobs.append(
                        dict(
                            name=name,
                            gpu=str(device["gpu"]),
                            repetition=repetition,
                            model=model["name"],
                            label=label,
                            command=command,
                            directory=str(directory),
                            hf_cache=model.get("hf_cache"),
                        )
                    )
        jobs.append(device_jobs)
    if not args.execute:
        print(
            json.dumps(
                {"sources": identities, "jobs": jobs, "budget_minutes": args.minutes},
                indent=2,
            )
        )
        return
    for device in plan["devices"]:
        if "uuid" in device:
            actual = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(device["gpu"]),
                    "--query-gpu=uuid",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
            if actual != device["uuid"]:
                raise ValueError(f"GPU {device['gpu']} identity changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for model in plan["models"]:
        (output / f"{model['name']}-cases.json").write_text(
            json.dumps(model["cases"], indent=2) + "\n"
        )
    deadline = time.monotonic() + args.minutes * 60
    stopped = threading.Event()

    def interrupted(signum, frame):
        stopped.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def run_device(device_jobs):
        completed = []
        journal = output / f"gpu{device_jobs[0]['gpu']}.json"
        for job in device_jobs:
            if stopped.is_set() or time.monotonic() >= deadline:
                raise RuntimeError("Campaign budget exhausted or another GPU failed")
            for name, repo in repositories.items():
                if source_state(repo)[2] != identities[name]:
                    stopped.set()
                    raise RuntimeError(f"{name} source changed")
            env = dict(os.environ)
            if job["hf_cache"]:
                env["HF_HUB_CACHE"] = job["hf_cache"]
            print(f"Starting {job['name']}", flush=True)
            with (output / f"{job['name']}.log").open("w") as log:
                process = subprocess.Popen(
                    job["command"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    while process.poll() is None:
                        if stopped.wait(timeout=1) or time.monotonic() >= deadline:
                            raise RuntimeError("Campaign deadline or peer failure")
                    if process.returncode:
                        raise RuntimeError(f"Failed block: {job['name']}")
                except BaseException:
                    stopped.set()
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=60)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    raise
            completed.append(job)
            journal.write_text(json.dumps(completed, indent=2) + "\n")
        return completed

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(run_device, device_jobs) for device_jobs in jobs]
        completed = [job for future in futures for job in future.result()]
    groups = {}
    for job in completed:
        directory = Path(job["directory"])
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["source_changed"]:
            raise RuntimeError(f"Invalid source: {directory}")
        for run in manifest["runs"]:
            result = json.loads((directory / run["name"] / "result.json").read_text())
            key = (job["gpu"], job["model"], run["name"])
            groups.setdefault(key, {}).setdefault(job["repetition"], {})[
                job["label"]
            ] = result
    rows = []
    for key, repetitions in sorted(groups.items()):
        ratios = [
            math.log(pair["candidate"]["output_throughput"])
            - math.log(pair["baseline"]["output_throughput"])
            for pair in repetitions.values()
        ]
        rows.append(
            dict(
                gpu=key[0],
                model=key[1],
                case=key[2],
                pairs=len(ratios),
                throughput_change_percent=100 * math.expm1(statistics.mean(ratios)),
                paired_log_ratios=ratios,
            )
        )
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Completed campaign: {output}", flush=True)


if __name__ == "__main__":
    main()
