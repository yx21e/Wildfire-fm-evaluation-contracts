#!/usr/bin/env python3
"""Write a SHA-256 manifest for selected release files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("release.sha256"))
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    lines = []
    for path in args.paths:
        if not path.is_file():
            raise SystemExit(f"not a file: {path}")
        lines.append(f"{sha256(path)}  {path.as_posix()}")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
