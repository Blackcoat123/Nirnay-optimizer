#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convex QP evidence: the Maros-Meszaros set against its reference optima, independently
verified.

    python bench/runners/fetch_maros_meszaros.py               # once
    python bench/runners/maros_meszaros.py --time-limit 60

An instance PASSES when the solver reports `optimal`, the objective matches the reference
to a relative 1e-6 (the README prints eight significant digits, so its own rounding is
below 5e-8), and tools/verify_solution.py - which shares no code with the solver - accepts
the written solution. The references are BPMPD's values from the distribution's README;
where a reference is itself inaccurate the row says so rather than being dropped.

Every run writes a CSV to bench/results/ with the instance sha256, both objectives, the
status, iterations, wall time, the algorithm that answered, the commit and the machine.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from machine import git_commit, machine_tag  # noqa: E402
from netlib import default_binary, run_one  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "maros"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

FIELDS = ["instance", "instance_sha256", "rows", "columns", "nonzeros",
          "quadratic_offdiagonal", "published_optimal", "status", "our_objective",
          "relative_error", "matched", "verified", "passed", "iterations", "wall_seconds",
          "algorithm", "time_limit", "solver_options", "git_commit", "machine",
          "timestamp_utc", "message"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="only instances with at most this many rows (0: all)")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    reference_path = DATA_DIR / "reference.json"
    if not reference_path.exists():
        raise SystemExit("no reference data; run bench/runners/fetch_maros_meszaros.py first")
    reference = json.loads(reference_path.read_text())["instances"]
    names = sorted(args.instances or reference)
    if args.max_rows > 0:
        names = [n for n in names if reference[n].get("rows", 0) <= args.max_rows]
    binary = args.binary or default_binary()
    commit, machine = git_commit(), machine_tag()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    options = " ".join(args.solver_option)
    print(f"solver {binary}\ncommit {commit}   machine {machine}   time limit "
          f"{args.time_limit:g}s{('   options ' + options) if options else ''}\n")
    print(f"{'instance':<12}{'status':<16}{'objective':>20}{'reference':>18}{'rel err':>10}"
          f"{'iters':>8}{'time':>9}  ver  result")

    rows = []
    passed = 0
    for name in names:
        entry = reference[name]
        path = DATA_DIR / f"{name}.qps"
        if not path.exists():
            continue
        result = run_one(binary, path, args.time_limit, not args.no_verify, args.solver_option)
        published = entry.get("published_optimal")
        objective = result.get("objective")
        relative = None
        if objective is not None and published is not None:
            relative = abs(objective - published) / max(1.0, abs(published))
        matched = relative is not None and relative <= 1e-6
        verified = result.get("verified")
        ok = result["status"] == "optimal" and matched and verified is not False
        passed += ok
        rows.append({
            "instance": name, "instance_sha256": entry.get("qps_sha256", ""),
            "rows": entry.get("rows", ""), "columns": entry.get("cols", ""),
            "nonzeros": entry.get("nonzeros", ""),
            "quadratic_offdiagonal": entry.get("quadratic_offdiagonal", ""),
            "published_optimal": published, "status": result["status"],
            "our_objective": objective, "relative_error": relative, "matched": int(matched),
            "verified": "" if verified is None else int(verified), "passed": int(ok),
            "iterations": result.get("iterations", ""),
            "wall_seconds": f"{result['wall_seconds']:.3f}",
            "algorithm": result.get("algorithm", ""), "time_limit": args.time_limit,
            "solver_options": options, "git_commit": commit, "machine": machine,
            "timestamp_utc": stamp, "message": (result.get("message") or "")[:300],
        })
        rel_text = f"{relative:.1e}" if relative is not None else "-"
        obj_text = f"{objective:.10g}" if objective is not None else "-"
        ver_text = "-" if verified is None else ("yes" if verified else "NO")
        print(f"{name:<12}{result['status']:<16}{obj_text:>20}{published:>18.8g}{rel_text:>10}"
              f"{str(result.get('iterations', '')):>8}{result['wall_seconds']:>8.2f}s  "
              f"{ver_text:<4} {'PASS' if ok else 'FAIL'}", flush=True)

    print(f"\n{passed}/{len(rows)} matched the reference to a relative 1e-6 and passed "
          f"independent verification")
    out = args.out or RESULTS_DIR / f"maros-meszaros-{commit}.csv"
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
