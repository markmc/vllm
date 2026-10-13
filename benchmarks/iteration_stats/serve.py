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

from run import capture, source_state


def main():
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
    args = parser.parse_args()
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
        "half",
        "--tensor-parallel-size",
        "1",
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
        "--no-enable-prefix-caching",
        "--async-scheduling" if args.async_mode == "on" else "--no-async-scheduling",
        "--enable-logging-iteration-details"
        if args.collection == "on"
        else "--no-enable-logging-iteration-details",
    ]
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
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "http_timeout_keep_alive_seconds": 120,
    }

    def save():
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(path):
        with opener.open(f"http://127.0.0.1:{args.port}/{path}", timeout=3) as response:
            return response.read().decode()

    try:
        get("health")
    except (OSError, urllib.error.URLError):
        pass
    else:
        raise RuntimeError(f"Port {args.port} already has a server")
    save()
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(
            server_command,
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 600
            while True:
                if server.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Server did not start; inspect server.log")
                try:
                    get("health")
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(1)
            (output / "metrics-before.txt").write_text(get("metrics"))
            cpu_samples = []
            ticks = os.sysconf("SC_CLK_TCK")
            with (output / "client.log").open("w") as client_log:
                client = subprocess.Popen(
                    client_command,
                    cwd=repo,
                    env=env,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                )
                started = time.monotonic()
                while True:
                    try:
                        fields = Path(f"/proc/{client.pid}/stat").read_text()
                        fields = fields.rsplit(")", 1)[1].split()
                        cpu_samples.append(
                            {
                                "elapsed_seconds": time.monotonic() - started,
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
                    raise subprocess.CalledProcessError(code, client_command)
            (output / "client-cpu.json").write_text(
                json.dumps(cpu_samples, indent=2) + "\n"
            )
            metrics = get("metrics")
            (output / "metrics-after.txt").write_text(metrics)
            preemptions = sum(
                float(line.rsplit(" ", 1)[1])
                for line in metrics.splitlines()
                if line.startswith("vllm:num_preemptions_total{")
            )
            result = json.loads((output / "result.json").read_text())
            manifest["preemptions"] = preemptions
            manifest["finished_revision"] = source_state(repo)[0]
            manifest["source_changed"] = source_state(repo)[2] != state[2]
            save()
            if result["completed"] != args.num_prompts or preemptions:
                raise RuntimeError("Run invalid: incomplete requests or preemptions")
            if manifest["source_changed"]:
                raise RuntimeError("Source changed during benchmark")
            print(f"Serving result saved: {output / 'result.json'}")
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


if __name__ == "__main__":
    main()
