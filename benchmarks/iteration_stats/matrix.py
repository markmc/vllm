# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run workload blocks with one warmed engine per comparison configuration."""

import argparse
import atexit
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from run import HERE, capture, source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases-file", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--cpu-list", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--async-modes", nargs="+", choices=["on", "off"], default=["on", "off"]
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["A", "B", "C", "B-repeat"],
        choices=["A", "B", "C", "B-repeat", "A-enabled"],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=75)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    telemetry_log = (output / "gpu-telemetry.csv").open("w")
    telemetry = subprocess.Popen(
        [
            "nvidia-smi",
            "-i",
            args.gpu,
            "--query-gpu=timestamp,uuid,clocks.sm,temperature.gpu,power.draw,utilization.gpu",
            "--format=csv",
            "--loop=5",
        ],
        stdout=telemetry_log,
        stderr=subprocess.STDOUT,
    )

    def stop_telemetry():
        telemetry.terminate()
        telemetry.wait(timeout=10)
        telemetry_log.close()

    atexit.register(stop_telemetry)
    cases = json.loads(args.cases_file.read_text())
    repos = {"patch": args.repo.resolve(), "base": args.base_repo.resolve()}
    states = {label: source_state(repo) for label, repo in repos.items()}
    hardware = {
        "gpu": capture(["nvidia-smi", "-q"]),
        "topology": capture(["nvidia-smi", "topo", "-m"]),
        "cpu": capture(["lscpu", "-e=CPU,CORE,SOCKET,NODE"]),
        "selected_uuid": capture(
            ["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader"]
        ).strip(),
    }
    packages = {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
    }
    artifacts = {
        label: {
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (repo / "vllm").rglob("*.so")
        }
        for label, repo in repos.items()
    }
    manifests = {}
    for case in cases:
        for mode in args.async_modes:
            directory = output / f"{case['name']}-async{mode}"
            directory.mkdir()
            settings = dict(
                vars(args),
                input_len=case["input_len"],
                output_len=case["output_len"],
                verify_delivery=False,
            )
            manifest = {
                "created_at": datetime.now(UTC).isoformat(),
                "arguments": settings,
                "hardware": hardware,
                "packages": packages,
                "native_artifacts": artifacts,
                "sources": {label: state[2] for label, state in states.items()},
                "runs": [],
                "engine_reuse": True,
            }
            manifests[case["name"], mode] = (directory, manifest)
            for label, state in states.items():
                (directory / f"{label}.revision").write_text(state[0] + "\n")
                (directory / f"{label}.patch").write_text(state[1])

    def save():
        for directory, manifest in manifests.values():
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2, default=str) + "\n"
            )

    calibration = {}
    save()
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpu,
        VLLM_LOGGING_LEVEL="WARNING",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONHASHSEED="0",
        HF_HUB_OFFLINE="1",
    )
    for repetition in range(args.repeats):
        for mode in args.async_modes:
            offset = (repetition // 2) % len(args.configs)
            order = args.configs[offset:] + args.configs[:offset]
            if repetition % 2:
                order.reverse()
            for label in order:
                source = "base" if label.startswith("A") else "patch"
                repo = repos[source]
                if source_state(repo)[2] != states[source][2]:
                    raise RuntimeError("Source changed during matrix")
                block = []
                entries = []
                for case in cases:
                    directory, manifest = manifests[case["name"], mode]
                    result = directory / f"r{repetition}-{label}.json"
                    count = calibration.get((case["name"], mode))
                    block.append(
                        dict(
                            case,
                            output_json=str(result),
                            num_iters=count or 1,
                            target_seconds=None if count else args.seconds,
                        )
                    )
                    entry = {
                        "label": label,
                        "batch": case["batch_size"],
                        "async": mode,
                        "repetition": repetition,
                        "result": str(result),
                        "source": states[source][2],
                        "started_at": datetime.now(UTC).isoformat(),
                    }
                    manifest["runs"].append(entry)
                    entries.append(entry)
                spec = output / f"block-r{repetition}-async{mode}-{label}.json"
                spec.write_text(json.dumps(block, indent=2) + "\n")
                command = [
                    "taskset",
                    "--cpu-list",
                    args.cpu_list,
                    sys.executable,
                    str(HERE / "latency.py"),
                    "--repo",
                    str(repo),
                    "--cases-file",
                    str(spec),
                    "--model",
                    args.model,
                    "--revision",
                    args.revision,
                    "--tokenizer-revision",
                    args.revision,
                    "--dtype",
                    "half",
                    "--tensor-parallel-size",
                    "1",
                    "--no-enable-prefix-caching",
                    "--disable-detokenize",
                    "--seed",
                    "0",
                    "--max-model-len",
                    "2048",
                    "--gpu-memory-utilization",
                    "0.6",
                    "--max-num-seqs",
                    "512",
                    "--max-num-batched-tokens",
                    "16384",
                    "--num-iters-warmup",
                    str(args.warmup),
                    "--output-json",
                    str(result),
                    "--async-scheduling" if mode == "on" else "--no-async-scheduling",
                    "--enable-logging-iteration-details"
                    if label in ("C", "A-enabled")
                    else "--no-enable-logging-iteration-details",
                ]
                for entry in entries:
                    entry["command"] = command
                save()
                print(
                    f"GPU {args.gpu} repetition {repetition} async {mode} {label}: "
                    f"{len(cases)} workloads",
                    flush=True,
                )
                env["PYTHONPATH"] = str(repo)
                log_path = output / f"block-r{repetition}-async{mode}-{label}.log"
                with log_path.open("w") as log:
                    process = subprocess.run(
                        command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                for case, entry in zip(cases, entries):
                    entry.update(
                        returncode=process.returncode,
                        source_changed=source_state(repo)[2] != states[source][2],
                        finished_at=datetime.now(UTC).isoformat(),
                    )
                    if process.returncode == 0:
                        raw = json.loads(Path(entry["result"]).read_text())
                        calibration[case["name"], mode] = len(raw["latencies"])
                        print(
                            f"  {case['name']}: {raw['avg_latency']:.6f} s, "
                            f"{raw['measured_seconds']:.1f} measured seconds",
                            flush=True,
                        )
                save()
                if process.returncode or any(e["source_changed"] for e in entries):
                    raise RuntimeError(f"Invalid block; inspect {log_path}")


if __name__ == "__main__":
    main()
