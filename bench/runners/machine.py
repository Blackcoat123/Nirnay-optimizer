# SPDX-License-Identifier: Apache-2.0
"""The machine tag every benchmark CSV records.

A timing is a property of the machine as much as of the solver, so the tag has to say which
machine: `Linux-x86_64` does not distinguish a laptop from a DGX node. The tag is, in order:

1. NIRNAY_MACHINE_TAG, when set - the way to name a Slurm slice precisely, for example
   `dgx-b200-mig1g45` when only a MIG slice was allocated;
2. otherwise `<system>-<arch>-<cpu model>`, plus `-<gpu name>` when nvidia-smi answers.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from functools import lru_cache


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or ""


def _gpu_name() -> str:
    if shutil.which("nvidia-smi") is None:
        return ""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    names = [line.strip() for line in out.splitlines() if line.strip()]
    return names[0] if names else ""


def _slug(text: str) -> str:
    text = re.sub(r"\((R|TM)\)|CPU|@.*$", "", text, flags=re.IGNORECASE)
    return re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-")


@lru_cache(maxsize=1)
def machine_tag() -> str:
    explicit = os.environ.get("NIRNAY_MACHINE_TAG", "").strip()
    if explicit:
        return explicit
    parts = [platform.system(), platform.machine()]
    cpu = _slug(_cpu_model())
    if cpu:
        parts.append(cpu)
    gpu = _slug(_gpu_name())
    if gpu:
        parts.append(gpu)
    return "-".join(parts)


@lru_cache(maxsize=1)
def git_commit() -> str:
    """Short commit hash, with "-dirty" appended when tracked files other than the tier
    manifests are modified - a result measured on uncommitted code must never be attributed
    to the commit it started from. The manifests (data/netlib/reference.json and
    data/mittelmann/reference.json) are rewritten by the fetch scripts as part of the
    runner's own workflow and say nothing about what was measured; untracked files are
    ignored for the same reason (fetched instances are untracked by design)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                capture_output=True, text=True, check=False)
        commit = result.stdout.strip() or "unknown"
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--",
             ".", ":!data/netlib/reference.json", ":!data/mittelmann/reference.json"],
            cwd=root, capture_output=True, text=True, check=False)
        if status.stdout.strip():
            commit += "-dirty"
        return commit
    except OSError:
        return "unknown"
