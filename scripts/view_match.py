#!/usr/bin/env python3
"""Serve the dynamic browser viewer for saved match JSON files."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thermo_chess.live_viewer import main


if __name__ == "__main__":
    main()
