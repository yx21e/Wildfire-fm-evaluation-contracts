#!/usr/bin/env python3
"""Create the CONUS FireWx-FM repaired probability candidate.

This is a thin wrapper around the source-probability seam-repair generator.
It exists so the all-US handoff has a CONUS-named entry point instead of the
older NE/central exploratory naming.
"""

from __future__ import annotations

from make_firewxfm_central_feathered_candidate import main


if __name__ == "__main__":
    raise SystemExit(main())
