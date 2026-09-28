#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rename the project everywhere it is named: paths, identifiers, headers, docs, scripts.

    python scripts/rename_project.py --from nirnay --to newname            # do it
    python scripts/rename_project.py --from nirnay --to newname --dry-run  # count only

Three spellings are replaced, each by the same spelling of the new name: lower case
(the namespace, `lib<name>.so`, the C API's `<name>_solve`), upper case (the CMake options
and the banner) and title case (the Python exception class). This script's own examples
name the project, so it rewrites them as well. Paths containing the name are moved (not with
`git mv`: nothing is staged, the working tree simply changes, and `git add -A` records the
moves as renames).

Files the rename must not touch are skipped: anything git ignores (build trees, fetched
benchmark instances), anything with a NUL byte (binaries), and anything over 20 MB. The
benchmark instances whose sha256 the evidence CSVs record carry no project name, so the
hashes stay true. Rebuild from a fresh build directory afterwards: the targets, the
library and the binary are renamed too.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 20 * 1024 * 1024


def candidate_files() -> list[Path]:
    """Tracked and untracked-but-not-ignored files that exist on disk."""
    listing = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=REPO_ROOT, capture_output=True, check=True).stdout.decode()
    files = []
    for name in listing.split("\0"):
        if not name:
            continue
        path = REPO_ROOT / name
        if path.is_file() and not path.is_symlink():
            files.append(path)
    return files


def spellings(old: str, new: str) -> list[tuple[str, str]]:
    return [(old.upper(), new.upper()), (old.capitalize(), new.capitalize()),
            (old.lower(), new.lower())]


def rewrite_contents(files: list[Path], pairs: list[tuple[str, str]], dry_run: bool) -> int:
    changed = 0
    for path in files:
        if path.stat().st_size > MAX_BYTES:
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new in pairs:
            updated = updated.replace(old, new)
        if updated != text:
            changed += 1
            if not dry_run:
                # Keep the exact line endings and bytes of everything that is not the name.
                path.write_bytes(updated.encode("utf-8"))
    return changed


def move_paths(files: list[Path], pairs: list[tuple[str, str]], dry_run: bool) -> int:
    """Move every file whose path relative to the repository contains the name."""
    moved = 0
    for path in sorted(files, key=lambda p: len(p.parts), reverse=True):
        relative = path.relative_to(REPO_ROOT).as_posix()
        target = relative
        for old, new in pairs:
            target = target.replace(old, new)
        if target == relative:
            continue
        moved += 1
        if dry_run:
            print(f"  {relative} -> {target}")
            continue
        destination = REPO_ROOT / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
    if not dry_run:
        # Remove directories the moves left empty.
        for directory in sorted(REPO_ROOT.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if (directory.is_dir() and ".git" not in directory.parts and
                    any(old in directory.name for old, _ in pairs) and
                    not any(directory.iterdir())):
                directory.rmdir()
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--from", dest="old", required=True, help="current name, any case")
    parser.add_argument("--to", dest="new", required=True, help="new name, any case")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.new.isidentifier():
        print(f"'{args.new}' is not usable as a C++ and Python identifier", file=sys.stderr)
        return 2
    pairs = spellings(args.old, args.new)
    files = candidate_files()
    changed = rewrite_contents(files, pairs, args.dry_run)
    moved = move_paths(files, pairs, args.dry_run)
    verb = "would change" if args.dry_run else "changed"
    print(f"{verb} {changed} file(s) and moved {moved} path(s): "
          f"{args.old} -> {args.new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
