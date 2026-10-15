# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Confirm one latency-sweep point using a local server and pinned client."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil
from run import capture, source_state


def main():
    wrapper_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--server-cpus", required=True)
    parser.add_argument("--client-cpus", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--num-prompts", type=int, default=1024)
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--async-mode", choices=["on", "off"], default="on")
    parser.add_argument("--collection", choices=["on", "off"], required=True)
    parser.add_argument("--cases-file", type=Path)
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--num-warmups", type=int)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    cases = json.loads(args.cases_file.read_text()) if args.cases_file else [{}]
    if not cases or not isinstance(cases, list):
        raise ValueError("Cases must be a nonempty JSON list")
    names = set()
    for index, case in enumerate(cases):
        name = case.setdefault("name", f"case{index}")
        if Path(name).name != name or name in (".", "..") or name in names:
            raise ValueError(f"Invalid or duplicate case name: {name}")
        names.add(name)
        for key in ("concurrency", "num_prompts", "input_len", "output_len"):
            case.setdefault(key, getattr(args, key))
            if case[key] < 1:
                raise ValueError(f"Require positive {key}")
        if case["input_len"] + case["output_len"] > args.max_model_len:
            raise ValueError("Case exceeds the server context limit")
        case.setdefault(
            "num_warmups",
            args.num_warmups
            if args.num_warmups is not None
            else max(32, 4 * case["concurrency"]),
        )
    from vllm.transformers_utils.repo_utils import hf_api

    tokenizer = hf_api().snapshot_download(
        args.model, revision=args.revision, local_files_only=True
    )
    repo, output = args.repo.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = source_state(repo)
    (output / "source.patch").write_text(state[1])
    env = dict(
        os.environ,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=args.gpu,
        VLLM_LOGGING_LEVEL="WARNING",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=str(repo),
        PYTHONHASHSEED="0",
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        VLLM_HTTP_TIMEOUT_KEEP_ALIVE="120",
    )
    cli = [str(args.python.absolute()), "-m", "vllm.entrypoints.cli.main"]
    server_command = [
        "taskset",
        "--cpu-list",
        args.server_cpus,
        *cli,
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--revision",
        args.revision,
        "--tokenizer-revision",
        args.revision,
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        "1",
        "--seed",
        "0",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--no-enable-prefix-caching",
        "--async-scheduling" if args.async_mode == "on" else "--no-async-scheduling",
        "--enable-logging-iteration-details"
        if args.collection == "on"
        else "--no-enable-logging-iteration-details",
    ]
    if args.language_model_only:
        server_command.append("--language-model-only")
    client_command = [
        "taskset",
        "--cpu-list",
        args.client_cpus,
        *cli,
        "bench",
        "serve",
        "--model",
        args.model,
        "--tokenizer",
        tokenizer,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.input_len),
        "--random-output-len",
        str(args.output_len),
        "--random-range-ratio",
        "0",
        "--random-prefix-len",
        "0",
        "--seed",
        "0",
        "--ignore-eos",
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(args.concurrency),
        "--num-prompts",
        str(args.num_prompts),
        "--num-warmups",
        str(max(32, 4 * args.concurrency)),
        "--percentile-metrics",
        "ttft,itl",
        "--metric-percentiles",
        "50,99",
        "--disable-tqdm",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(output),
        "--result-filename",
        "result.json",
    ]
    manifest = {
        "args": vars(args),
        "source": state[2],
        "revision": state[0],
        "server_command": server_command,
        "client_command": client_command,
        "hardware": capture(["nvidia-smi", "-q"]),
        "topology": capture(["nvidia-smi", "topo", "-m"]),
        "selected_gpu_uuid": capture(
            ["nvidia-smi", "-i", args.gpu, "--query-gpu=uuid", "--format=csv,noheader"]
        ).strip(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "http_timeout_keep_alive_seconds": 120,
        "runs": [],
    }

    def save():
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(path):
        with opener.open(f"http://127.0.0.1:{args.port}/{path}", timeout=3) as response:
            return response.read().decode()

    def owns_listener():
        try:
            parent = psutil.Process(server.pid)
            return any(
                connection.status == psutil.CONN_LISTEN
                and connection.laddr.port == args.port
                for process in [parent, *parent.children(recursive=True)]
                for connection in process.net_connections(kind="tcp")
            )
        except psutil.NoSuchProcess:
            return False

    try:
        get("health")
    except (OSError, urllib.error.URLError):
        pass
    else:
        raise RuntimeError(f"Port {args.port} already has a server")
    save()
    started = time.monotonic()
    manifest["pre_server_seconds"] = started - wrapper_started
    telemetry_log = (output / "gpu-telemetry.csv").open("w")
    telemetry = subprocess.Popen(
        [
            "nvidia-smi",
            "-i",
            args.gpu,
            "--query-gpu=timestamp,uuid,clocks.sm,power.draw,utilization.gpu,memory.used",
            "--format=csv",
            "--loop=1",
        ],
        stdout=telemetry_log,
        stderr=subprocess.STDOUT,
    )
    with (output / "server.log").open("w") as log:
        client = None
        server = subprocess.Popen(
            server_command,
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        manifest["server_pid"] = server.pid
        save()
        try:
            deadline = time.monotonic() + 600
            while True:
                if server.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Server did not start; inspect server.log")
                try:
                    get("health")
                    if owns_listener():
                        break
                    raise RuntimeError("Health endpoint belongs to another process")
                except (OSError, urllib.error.URLError):
                    time.sleep(1)
            manifest["server_ready_seconds"] = time.monotonic() - started
            save()
            print(
                f"Server ready after {manifest['server_ready_seconds']:.1f}s",
                flush=True,
            )
            for case in cases:
                case_output = output / case["name"] if args.cases_file else output
                case_output.mkdir(exist_ok=True)
                command = list(client_command)
                for flag, value in (
                    ("--random-input-len", case["input_len"]),
                    ("--random-output-len", case["output_len"]),
                    ("--max-concurrency", case["concurrency"]),
                    ("--num-prompts", case["num_prompts"]),
                    ("--num-warmups", case["num_warmups"]),
                    ("--result-dir", case_output),
                ):
                    command[command.index(flag) + 1] = str(value)
                run = dict(case, command=command, started_at=time.time())
                manifest["runs"].append(run)
                save()
                (case_output / "metrics-before.txt").write_text(get("metrics"))
                cpu_samples = []
                ticks = os.sysconf("SC_CLK_TCK")
                client_started = time.monotonic()
                with (case_output / "client.log").open("w") as client_log:
                    client = subprocess.Popen(
                        command,
                        cwd=repo,
                        env=env,
                        stdout=client_log,
                        stderr=subprocess.STDOUT,
                    )
                    while True:
                        try:
                            fields = Path(f"/proc/{client.pid}/stat").read_text()
                            fields = fields.rsplit(")", 1)[1].split()
                            cpu_samples.append(
                                {
                                    "elapsed_seconds": (
                                        time.monotonic() - client_started
                                    ),
                                    "cpu_seconds": (int(fields[11]) + int(fields[12]))
                                    / ticks,
                                }
                            )
                        except FileNotFoundError:
                            pass
                        try:
                            code = client.wait(timeout=1)
                            break
                        except subprocess.TimeoutExpired:
                            continue
                    if code:
                        raise subprocess.CalledProcessError(code, command)
                (case_output / "client-cpu.json").write_text(
                    json.dumps(cpu_samples, indent=2) + "\n"
                )
                metrics = get("metrics")
                (case_output / "metrics-after.txt").write_text(metrics)
                preemptions = sum(
                    float(line.rsplit(" ", 1)[1])
                    for line in metrics.splitlines()
                    if line.startswith("vllm:num_preemptions_total{")
                )
                result = json.loads((case_output / "result.json").read_text())
                if result["completed"] != case["num_prompts"] or result["failed"]:
                    raise RuntimeError("Run invalid: incomplete requests")
                for phase in ("input", "output"):
                    if result[f"total_{phase}_tokens"] != (
                        case["num_prompts"] * case[f"{phase}_len"]
                    ):
                        raise RuntimeError(f"Run invalid: unexpected {phase} length")
                if preemptions:
                    raise RuntimeError("Run invalid: preemptions")
                run.update(
                    finished_at=time.time(),
                    client_wall_seconds=time.monotonic() - client_started,
                    measured_seconds=result["duration"],
                    output_throughput=result["output_throughput"],
                    preemptions=preemptions,
                )
                clock_offset = time.time() - time.perf_counter()
                run["measurement_started_at"] = (
                    min(result["start_times"]) + clock_offset
                )
                run["measurement_finished_at"] = (
                    max(
                        start + latency
                        for start, latency in zip(
                            result["start_times"], result["latencies"]
                        )
                    )
                    + clock_offset
                )
                manifest["preemptions"] = preemptions
                save()
                print(
                    f"{case['name']}: {result['output_throughput']:.1f} output tok/s, "
                    f"{result['duration']:.1f}s measured, "
                    f"{run['client_wall_seconds']:.1f}s client wall time",
                    flush=True,
                )
            manifest["finished_revision"] = source_state(repo)[0]
            manifest["source_changed"] = source_state(repo)[2] != state[2]
            manifest["total_wall_seconds"] = time.monotonic() - started
            save()
            if manifest["source_changed"]:
                raise RuntimeError("Source changed during benchmark")
            print(f"Serving block saved: {output}", flush=True)
        finally:
            if client is not None and client.poll() is None:
                client.terminate()
                try:
                    client.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    client.kill()
                    client.wait()
            telemetry.terminate()
            telemetry.wait(timeout=10)
            telemetry_log.close()
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            manifest["wrapper_wall_seconds"] = time.monotonic() - wrapper_started
            save()


if __name__ == "__main__":
    main()
