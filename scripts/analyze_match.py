#!/usr/bin/env python3
"""Inspect a saved match and print move-landscape tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FEATURE_COLUMNS = [
    "material",
    "center",
    "development",
    "castling",
    "king_safety",
    "king_pressure",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("match_json", nargs="?", default="data/games/json/thermo_match.json")
    parser.add_argument("--ply", type=int, default=1)
    parser.add_argument(
        "--sort", choices=["probability", "static_after", "phi"], default="probability"
    )
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    data = json.loads(Path(args.match_json).read_text(encoding="utf-8"))
    entry = next(item for item in data["plies"] if int(item["ply"]) == args.ply)
    moves = sorted(
        entry["current_landscape"]["moves"],
        key=lambda item: item[args.sort],
        reverse=True,
    )

    print(f"Ply {args.ply}: {entry['row']['side']} played {entry['row']['san']}")
    header = (
        f"{'move':<10} {'phi':>8} {'p(move|B)':>12} {'E(B_move)':>11} "
        + " ".join(f"{name[:8]:>8}" for name in FEATURE_COLUMNS)
    )
    print(header)
    print("-" * len(header))
    for move in moves[: args.limit]:
        features = move.get("features", {})
        feature_values = " ".join(
            f"{float(features.get(name, 0.0)):>8.3f}" for name in FEATURE_COLUMNS
        )
        print(
            f"{move['san']:<10} "
            f"{move['phi']:>8.3f} "
            f"{move['probability']:>12.5f} "
            f"{move['static_after']:>11.3f} "
            f"{feature_values}"
        )


if __name__ == "__main__":
    main()
