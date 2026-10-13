# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze base and draft sources with identical installed native artifacts."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from run import source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    revision, patch, fingerprint = source_state(repo)
    base_revision = subprocess.check_output(
        ["git", "rev-parse", args.base_revision], cwd=repo, text=True
    ).strip()
    artifacts = []
    for ignored in (False, True):
        command = ["git", "ls-files", "-z", "--others", "--exclude-standard"]
        if ignored:
            command.append("--ignored")
        command.extend(["--", "vllm"])
        paths = subprocess.check_output(command, cwd=repo).decode().split("\0")
        for name in filter(None, paths):
            path = Path(name)
            if path.suffix in (".so", ".py") or path.name == "vllm-rs":
                artifacts.append(path)
    artifact_hashes = {}
    for label in ("base", "patch"):
        target = output / label
        subprocess.run(
            ["git", "clone", "--shared", "--no-checkout", str(repo), str(target)],
            check=True,
        )
        checkout = base_revision if label == "base" else revision
        subprocess.run(
            ["git", "checkout", "--detach", checkout], cwd=target, check=True
        )
        if label == "patch" and patch:
            subprocess.run(
                ["git", "apply", "-"], cwd=target, input=patch, text=True, check=True
            )
        for relative in artifacts:
            src, dest = repo / relative, target / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            if label == "base":
                artifact_hashes[str(relative)] = digest
            elif artifact_hashes[str(relative)] != digest:
                raise RuntimeError(f"Installed artifact changed during snapshot: {src}")
    if source_state(repo)[2] != fingerprint:
        raise RuntimeError("Draft changed while snapshotting; create another snapshot")
    (output / "snapshot.json").write_text(
        json.dumps(
            {
                "source": str(repo),
                "base_revision": base_revision,
                "patch_revision": revision,
                "source_fingerprint": fingerprint,
                "artifact_sha256": artifact_hashes,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Frozen benchmark sources: {output}")


if __name__ == "__main__":
    main()
