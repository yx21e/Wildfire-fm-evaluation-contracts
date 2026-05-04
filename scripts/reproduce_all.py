#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    run([sys.executable, "scripts/build_tables.py"])
    run([sys.executable, "scripts/build_figures.py"])
    run([sys.executable, "scripts/audit_release.py"])
    print("Rebuilt tables and figures under paper_outputs/ and passed release audit.")


if __name__ == "__main__":
    main()
