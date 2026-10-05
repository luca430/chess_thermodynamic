#!/usr/bin/env python3
"""Compare immediate response-ensemble values with deep-search values."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

DEFAULT_COVARIANCE_DIR = Path("data/analysis/covariance")
DEFAULT_DIVERGENCE_DIR = Path("data/analysis/deep_search_divergence")

FLOAT_TOLERANCE = 1e-9

OUTPUT_FIELDS = [
    "game",
    "ply",
    "move_number",
    "side",
    "player",
    "candidate_san",
    "candidate_uci",
    "candidate_probability",
    "candidate_was_actual_move",
    "selected_actual_move",
    "number_of_responses",
    "retained_probability_mass",
    "retained_response_count",
    "mean_static_evaluation",
    "variance_static_evaluation",
    "terminal_u",
    "branch_value",
    "g_tilde",
    "q_tilde",
    "w_tilde",
    "a_tilde",
    "terminal_minus_mean_static",
    "abs_terminal_minus_mean_static",
]


class DivergenceAnalysisError(ValueError):
    """Raised when covariance rows cannot be matched safely to game JSON."""


def _as_float(value: Any, *, required: bool = False, field: str = "") -> float | None:
    if value in (None, ""):
        if required:
            raise DivergenceAnalysisError(f"missing required field {field}")
        return None
    return float(value)


def _as_int(value: Any, *, field: str) -> int:
    if value in (None, ""):
        raise DivergenceAnalysisError(f"missing required field {field}")
    return int(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _game_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(item for item in path.glob("*.json") if item.is_file()))
        else:
            files.append(path)
    return files


def read_covariance_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _deepened_candidates(game_json: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    candidates: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in game_json.get("plies", []):
        ply = _as_int(entry.get("ply", entry.get("row", {}).get("ply")), field="ply")
        search = entry.get("search_result") or entry.get("actual_search_result") or {}
        for index, move in enumerate(search.get("moves", [])):
            if not (move.get("selected_for_refinement") or move.get("actually_deepened")):
                continue
            uci = str(move.get("uci", ""))
            key = (ply, uci)
            if key in candidates:
                raise DivergenceAnalysisError(f"duplicate deepened JSON candidate key {key}")
            enriched = dict(move)
            enriched["_search_order"] = index
            enriched["_ply"] = ply
            enriched["_row"] = entry.get("row", {})
            candidates[key] = enriched
    return candidates


def _covariance_key(row: dict[str, Any]) -> tuple[int, str]:
    return (
        _as_int(row.get("ply"), field="ply"),
        str(row.get("candidate_uci", "")),
    )


def _validate_duplicate_covariance_keys(rows: list[dict[str, Any]]) -> None:
    seen: set[tuple[int, str]] = set()
    for row in rows:
        key = _covariance_key(row)
        if key in seen:
            raise DivergenceAnalysisError(f"duplicate covariance candidate key {key}")
        seen.add(key)


def _validate_match(row: dict[str, Any], move: dict[str, Any]) -> None:
    row_san = str(row.get("candidate_san", ""))
    move_san = str(move.get("san", ""))
    if row_san and move_san and row_san != move_san:
        raise DivergenceAnalysisError(
            f"SAN mismatch for {row.get('ply')} {row.get('candidate_uci')}: "
            f"covariance={row_san!r}, json={move_san!r}"
        )

    row_probability = _as_float(row.get("candidate_probability"))
    move_probability = _as_float(move.get("probability"))
    if (
        row_probability is not None
        and move_probability is not None
        and not math.isclose(row_probability, move_probability, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE)
    ):
        raise DivergenceAnalysisError(
            f"candidate probability mismatch for {row.get('ply')} {row.get('candidate_uci')}: "
            f"covariance={row_probability}, json={move_probability}"
        )

    row_count = row.get("number_of_responses")
    json_count = move.get("retained_response_count")
    if row_count not in (None, "") and json_count not in (None, ""):
        if int(row_count) != int(json_count):
            raise DivergenceAnalysisError(
                f"response count mismatch for {row.get('ply')} {row.get('candidate_uci')}: "
                f"covariance={row_count}, json={json_count}"
            )


def divergence_rows(
    covariance_rows: list[dict[str, Any]],
    game_json: dict[str, Any],
    *,
    game: str,
) -> list[dict[str, Any]]:
    _validate_duplicate_covariance_keys(covariance_rows)
    candidates = _deepened_candidates(game_json)
    rows: list[dict[str, Any]] = []
    for covariance in covariance_rows:
        key = _covariance_key(covariance)
        if key not in candidates:
            raise DivergenceAnalysisError(f"no matching deepened JSON candidate for key {key}")
        move = candidates[key]
        _validate_match(covariance, move)

        mean_static = _as_float(
            covariance.get("mean_static_evaluation"),
            required=True,
            field="mean_static_evaluation",
        )
        terminal_u = _as_float(move.get("terminal_u"), required=True, field="terminal_u")
        g_tilde = _as_float(move.get("g_tilde"), required=True, field="g_tilde")
        assert mean_static is not None
        assert terminal_u is not None

        delta = terminal_u - mean_static
        row = {
            "game": covariance.get("game") or game,
            "ply": key[0],
            "move_number": covariance.get("move_number"),
            "side": covariance.get("side"),
            "player": covariance.get("player"),
            "candidate_san": covariance.get("candidate_san") or move.get("san"),
            "candidate_uci": key[1],
            "candidate_probability": covariance.get("candidate_probability", move.get("probability")),
            "candidate_was_actual_move": covariance.get("candidate_was_actual_move"),
            "selected_actual_move": covariance.get("selected_actual_move"),
            "number_of_responses": covariance.get("number_of_responses"),
            "retained_probability_mass": move.get("retained_probability_mass"),
            "retained_response_count": move.get("retained_response_count"),
            "mean_static_evaluation": mean_static,
            "variance_static_evaluation": _as_float(covariance.get("variance_static_evaluation")),
            "terminal_u": terminal_u,
            "branch_value": _as_float(move.get("branch_value")),
            "g_tilde": g_tilde,
            "q_tilde": _as_float(move.get("q_tilde")),
            "w_tilde": _as_float(move.get("w_tilde")),
            "a_tilde": _as_float(move.get("a_tilde")),
            "terminal_minus_mean_static": delta,
            "abs_terminal_minus_mean_static": abs(delta),
            "_search_order": move.get("_search_order", 0),
        }
        rows.append(row)
    return rows


def analyze_pair(covariance_csv: Path, game_json: Path) -> list[dict[str, Any]]:
    return divergence_rows(
        read_covariance_rows(covariance_csv),
        json.loads(game_json.read_text(encoding="utf-8")),
        game=game_json.stem,
    )


def covariance_path_for_game(game_json: Path, covariance_dir: Path) -> Path:
    return covariance_dir / f"{game_json.stem}.csv"


def _is_directory_output(path: Path, input_count: int) -> bool:
    return path.exists() and path.is_dir() or path.suffix == "" or input_count > 1


def output_path_for_game(base: Path | None, game_json: Path, input_count: int) -> Path:
    if base is None:
        return DEFAULT_DIVERGENCE_DIR / f"{game_json.stem}.csv"
    if _is_directory_output(base, input_count):
        return base / f"{game_json.stem}.csv"
    return base


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rank_by_abs_divergence(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: float(row["abs_terminal_minus_mean_static"]),
        reverse=True,
    )[:limit]


def rows_for_ply(rows: list[dict[str, Any]], ply: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if int(row["ply"]) == ply]
    return sorted(
        selected,
        key=lambda row: (int(row.get("_search_order", 0)), -float(row.get("candidate_probability") or 0.0)),
    )


def _print_table(rows: list[dict[str, Any]], fields: list[str]) -> None:
    if not rows:
        print("No matching rows.")
        return
    widths = {
        field: max(len(field), *(len(str(row.get(field, ""))) for row in rows))
        for field in fields
    }
    print("  ".join(field.ljust(widths[field]) for field in fields))
    print("  ".join("-" * widths[field] for field in fields))
    for row in rows:
        print("  ".join(str(row.get(field, "")).ljust(widths[field]) for field in fields))


def print_top(rows: list[dict[str, Any]], limit: int) -> None:
    fields = [
        "ply",
        "candidate_san",
        "candidate_uci",
        "candidate_probability",
        "candidate_was_actual_move",
        "mean_static_evaluation",
        "terminal_u",
        "terminal_minus_mean_static",
        "variance_static_evaluation",
        "g_tilde",
    ]
    _print_table(rank_by_abs_divergence(rows, limit), fields)


def print_ply(rows: list[dict[str, Any]], ply: int) -> None:
    fields = [
        "candidate_san",
        "candidate_uci",
        "candidate_probability",
        "candidate_was_actual_move",
        "number_of_responses",
        "mean_static_evaluation",
        "variance_static_evaluation",
        "terminal_u",
        "g_tilde",
        "q_tilde",
        "w_tilde",
        "a_tilde",
        "terminal_minus_mean_static",
    ]
    _print_table(rows_for_ply(rows, ply), fields)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare <E>_m from retained response ensembles with terminal_u from "
            "the deeper search. terminal_minus_mean_static = terminal_u - "
            "mean_static_evaluation; large absolute values identify branches where "
            "recursive search substantially changes the immediate static assessment."
        )
    )
    parser.add_argument("game_json", nargs="+", type=Path, help="Detailed game JSON file(s) or directories.")
    parser.add_argument(
        "--covariance-csv",
        type=Path,
        default=None,
        help="Covariance CSV for a single game. Defaults to data/analysis/covariance/<game>.csv.",
    )
    parser.add_argument(
        "--covariance-dir",
        type=Path,
        default=DEFAULT_COVARIANCE_DIR,
        help="Directory containing covariance CSV files named after game JSON stems.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV file or directory. Defaults to data/analysis/deep_search_divergence/<game>.csv.",
    )
    parser.add_argument("--top", type=int, default=None, help="Print the top N rows ranked by absolute divergence.")
    parser.add_argument("--ply", type=int, default=None, help="Print all rows for one ply.")
    args = parser.parse_args()

    game_files = _game_files(args.game_json)
    if args.covariance_csv is not None and len(game_files) != 1:
        raise SystemExit("--covariance-csv can only be used with one game JSON")

    all_rows: list[dict[str, Any]] = []
    wrote_paths: list[Path] = []
    try:
        for game_json in game_files:
            covariance_csv = args.covariance_csv or covariance_path_for_game(
                game_json, args.covariance_dir
            )
            if not covariance_csv.exists():
                raise DivergenceAnalysisError(f"missing covariance CSV: {covariance_csv}")
            rows = analyze_pair(covariance_csv, game_json)
            output_path = output_path_for_game(args.output, game_json, len(game_files))
            write_rows(rows, output_path)
            print(f"Wrote {output_path}", file=sys.stderr)
            wrote_paths.append(output_path)
            all_rows.extend(rows)
    except DivergenceAnalysisError as exc:
        raise SystemExit(str(exc)) from exc

    if args.top is not None:
        print_top(all_rows, args.top)
    if args.ply is not None:
        print_ply(all_rows, args.ply)


if __name__ == "__main__":
    main()
