#!/usr/bin/env python3
"""Build or repair the live 5 km FireWx-FM input stack.

This is the production-facing entry point for the 16-channel FireWx-FM stack.
It applies the corrected static-layer path used by the staticfix CONUS model:

1. start from a live 16-channel stack produced by the daily HRRR pipeline,
2. replace channels 11-15 from an audited 5 km static cache or raw static
   rasters on the same EPSG:5070 grid, and
3. write a corrected NumPy/TIFF stack plus a JSON audit summary.

The implementation lives in ``make_firewxfm_current_static_corrected_stack`` so
older experiment commands remain valid. This wrapper gives production a stable
script name that can replace the older ``build_5km_input_stack.py``.
"""

from __future__ import annotations

from make_firewxfm_current_static_corrected_stack import main


if __name__ == "__main__":
    raise SystemExit(main())
