# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run paired phase-1 latency comparisons on one GPU per invocation."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def capture(command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True)


def source_state(repo):
    revision = capture(["git", "rev-parse", "HEAD"], repo).strip()
    tree = capture(["git", "rev-parse", "HEAD:vllm"], repo).strip()
    patch = capture(["git", "diff", "HEAD", "--", "vllm"], repo)
    return revision, patch, hashlib.sha256((tree + patch).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base-repo", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="Physical GPU index or UUID")
    parser.add_argument("--cpu-list", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", required=True, help="Immutable model commit")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128, 512])
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument(
        "--async-modes", nargs="+", choices=["on", "off"], default=["on", "off"]
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["B", "C"],
        choices=["A", "B", "C", "B-repeat", "A-enabled"],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--verify-delivery",
        action="store_true",
        help="Instrument delivery; these results are untimed checks",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(c.startswith("A") for c in args.configs) and args.base_repo is None:
        parser.error("A configurations require --base-repo")
    if min(args.repeats, args.num_iters, args.warmup, *args.batch_sizes) < 1:
        parser.error("Repetition, batch and warmup counts must be positive")
    if len(set(args.configs)) != len(args.configs):
        parser.error("Use B-repeat for unchanged-versus-unchanged comparisons")
    output = args.output_dir.resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError("Choose a new output directory for each sweep")
    output.mkdir(parents=True, exist_ok=True)
    repos = {"patch": args.repo.resolve()}
    if args.base_repo:
        repos["base"] = args.base_repo.resolve()
    states = {name: source_state(repo) for name, repo in repos.items()}
    for name, (revision, patch, _) in states.items():
        (output / f"{name}.patch").write_text(patch)
        (output / f"{name}.revision").write_text(revision + "\n")
    hardware = {}
    if not args.dry_run:
        for name, command in {
            "gpu": ["nvidia-smi", "-q"],
            "topology": ["nvidia-smi", "topo", "-m"],
            "cpu": ["lscpu", "-e=CPU,CORE,SOCKET,NODE"],
        }.items():
            hardware[name] = capture(command)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "arguments": vars(args),
        "hardware": hardware,
        "sources": {name: state[2] for name, state in states.items()},
        "packages": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
        },
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("VLLM_", "CUDA_", "OMP_", "MKL_"))
        },
        "runs": [],
    }
    manifest["native_artifacts"] = {
        name: {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (repo / "vllm").rglob("*.so")
        }
        for name, repo in repos.items()
    }

    def save_manifest():
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )

    save_manifest()
    for batch in args.batch_sizes:
        for async_mode in args.async_modes:
            for repetition in range(args.repeats):
                offset = (repetition // 2) % len(args.configs)
                order = args.configs[offset:] + args.configs[:offset]
                if repetition % 2:
                    order = list(reversed(order))
                for label in order:
                    source = "base" if label.startswith("A") else "patch"
                    repo = repos[source]
                    stem = (
                        f"gpu{args.gpu}-b{batch}-async{async_mode}"
                        f"-r{repetition}-{label}"
                    )
                    result = output / f"{stem}.json"
                    if result.exists():
                        raise FileExistsError(result)
                    command = [
                        "taskset",
                        "--cpu-list",
                        args.cpu_list,
                        str(args.python.absolute()),
                        str(HERE / "latency.py"),
                        "--repo",
                        str(repo),
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
                        "--input-len",
                        str(args.input_len),
                        "--output-len",
                        str(args.output_len),
                        "--batch-size",
                        str(batch),
                        "--num-iters",
                        str(args.num_iters),
                        "--num-iters-warmup",
                        str(args.warmup),
                        "--output-json",
                        str(result),
                        "--async-scheduling"
                        if async_mode == "on"
                        else "--no-async-scheduling",
                    ]
                    if label in ("C", "A-enabled"):
                        command.append("--enable-logging-iteration-details")
                    else:
                        command.append("--no-enable-logging-iteration-details")
                    if args.verify_delivery:
                        command.append("--verify-delivery")
                    env = dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=args.gpu,
                        VLLM_LOGGING_LEVEL="WARNING",
                        OMP_NUM_THREADS="1",
                        MKL_NUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1",
                        TOKENIZERS_PARALLELISM="false",
                        PYTHONHASHSEED="0",
                        PYTHONPATH=str(repo),
                        HF_HUB_OFFLINE="1",
                    )
                    entry = {
                        "started_at": datetime.now(UTC).isoformat(),
                        "label": label,
                        "batch": batch,
                        "async": async_mode,
                        "repetition": repetition,
                        "command": command,
                        "result": str(result),
                        "source": states[source][2],
                        "environment": {
                            k: env[k]
                            for k in (
                                "CUDA_VISIBLE_DEVICES",
                                "VLLM_LOGGING_LEVEL",
                                "OMP_NUM_THREADS",
                                "MKL_NUM_THREADS",
                                "OPENBLAS_NUM_THREADS",
                                "TOKENIZERS_PARALLELISM",
                                "PYTHONHASHSEED",
                                "PYTHONPATH",
                                "HF_HUB_OFFLINE",
                            )
                        },
                    }
                    manifest["runs"].append(entry)
                    save_manifest()
                    print(shlex.join(command), flush=True)
                    if args.dry_run:
                        continue
                    if source_state(repo)[2] != states[source][2]:
                        raise RuntimeError(
                            "Source changed during sweep; start a new run"
                        )
                    with (output / f"{stem}.log").open("w") as log:
                        process = subprocess.run(
                            command,
                            cwd=repo,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                    entry["returncode"] = process.returncode
                    entry["finished_at"] = datetime.now(UTC).isoformat()
                    entry["finished_revision"] = source_state(repo)[0]
                    entry["source_changed"] = source_state(repo)[2] != states[source][2]
                    save_manifest()
                    if process.returncode or entry["source_changed"]:
                        raise RuntimeError(f"Run invalid; inspect {stem}.log")
                    raw = json.loads(result.read_text())
                    print(
                        f"{stem}: {raw['avg_latency']:.6f} s, "
                        f"{raw['output_token_throughput']:.1f} output tok/s, "
                        f"{raw['measured_seconds']:.1f} measured seconds",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
