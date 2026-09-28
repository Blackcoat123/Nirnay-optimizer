#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch the Maros-Meszaros convex QP test set and its reference optima.

PS26119 names QPLIB "for quadratic programming where applicable"; the Maros-Meszaros set is
the standard convex-QP benchmark that QPLIB's convex part grew out of, and the one every
convex QP solver reports against:

    I. Maros, Cs. Meszaros, "A repository of convex quadratic programming problems",
    Optimization Methods and Software 11-12 (1999), 671-681.

The instances come from the authors' own distribution,

    https://www.doc.ic.ac.uk/~im/QPDATA1.ZIP   (76 problems from CUTE)
    https://www.doc.ic.ac.uk/~im/QPDATA2.ZIP   (46 from the Brunel group)
    https://www.doc.ic.ac.uk/~im/QPDATA3.ZIP   (16 from miscellaneous sources)

as QPS files (MPS plus a QUADOBJ section), and the reference optimum of each is PARSED from
the distribution's own 00README.QP table (column OPT: "solution value obtained by the default
settings of BPMPD solver"), never typed in. The files are DOS text; they are written with
LF line endings here so their sha256 is the same on every platform, and the manifest
records the sha256 of every archive and every decoded file.

    python bench/runners/fetch_maros_meszaros.py            # all 138
    python bench/runners/fetch_maros_meszaros.py --force    # re-download
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "maros"
BASE_URL = "https://www.doc.ic.ac.uk/~im/"
ARCHIVES = ["QPDATA1.ZIP", "QPDATA2.ZIP", "QPDATA3.ZIP"]
README = "00README.QP"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, attempts: int = 4) -> bytes:
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except OSError as error:
            if attempt == attempts:
                raise
            wait = 2 ** (attempt - 1)
            print(f"  {url}: {error}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def parse_readme(text: str) -> dict[str, dict]:
    """The problem table: NAME M N NZ QN QNZ OPT, one problem per line."""
    table: dict[str, dict] = {}
    row = re.compile(r"^\s*([A-Za-z0-9_\-]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
                     r"([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)\s*$")
    for line in text.splitlines():
        match = row.match(line)
        if not match:
            continue
        name = match.group(1).lower()
        table[name] = {
            "rows": int(match.group(2)),
            "cols": int(match.group(3)),
            "nonzeros": int(match.group(4)),
            "quadratic_columns": int(match.group(5)),
            "quadratic_offdiagonal": int(match.group(6)),
            "published_optimal": float(match.group(7).replace("D", "e").replace("d", "e")),
        }
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="re-download present files")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    readme_bytes = download(BASE_URL + README)
    (DATA_DIR / README).write_bytes(readme_bytes)
    table = parse_readme(readme_bytes.decode("latin-1"))
    print(f"{README}: {len(table)} problems with a reference optimum")

    manifest = {"source": BASE_URL, "readme_sha256": sha256_bytes(readme_bytes),
                "archives": {}, "instances": {}}
    for archive in ARCHIVES:
        print(f"fetching {BASE_URL + archive}")
        data = download(BASE_URL + archive)
        manifest["archives"][archive] = {"sha256": sha256_bytes(data), "bytes": len(data)}
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            for member in bundle.namelist():
                if not member.lower().endswith(".qps"):
                    continue
                name = Path(member).stem.lower()
                target = DATA_DIR / f"{name}.qps"
                raw = bundle.read(member)
                text = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                if args.force or not target.exists():
                    target.write_bytes(text)
                entry = {"archive": archive, "qps_sha256": sha256_bytes(text),
                         "qps_bytes": len(text)}
                # The README spells cvxqp1_l as cvxqp1l; match on the name without "_".
                entry.update(table.get(name, table.get(name.replace("_", ""), {})))
                manifest["instances"][name] = entry

    have = {name.replace("_", "") for name in manifest["instances"]} | set(manifest["instances"])
    missing = sorted(set(table) - have)
    if missing:
        print(f"note: {len(missing)} README entries have no file: {', '.join(missing)}")
    with (DATA_DIR / "reference.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {DATA_DIR / 'reference.json'}: {len(manifest['instances'])} instance(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
