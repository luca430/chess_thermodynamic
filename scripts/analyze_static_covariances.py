#!/usr/bin/env python3
"""Analyze response-ensemble covariances of static evaluator components."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

COMPONENT_ORDER = (
    "material",
    "pawn_structure",
    "mobility",
    "center",
    "king_safety",
)

PROBABILITY_TOLERANCE = 1e-9
DEFAULT_COVARIANCE_DIR = Path("data/analysis/covariance")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _game_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(item for item in path.glob("*.json") if item.is_file()))
        else:
            files.append(path)
    return files


def normalize_probabilities(
    responses: list[dict[str, Any]],
    *,
    retained_probability_mass: float | None = None,
    tolerance: float = PROBABILITY_TOLERANCE,
) -> tuple[list[float], float, bool]:
    raw = [
        _as_float(response.get("conditional_probability"))
        for response in responses
    ]
    if all(value is not None for value in raw):
        probabilities = [float(value) for value in raw]
        total = sum(probabilities)
        if math.isclose(total, 1.0, rel_tol=tolerance, abs_tol=tolerance):
            return probabilities, total, False
        if total > 0.0:
            return [value / total for value in probabilities], total, True

    if retained_probability_mass is not None and retained_probability_mass > 0.0:
        probabilities = [
            float(response.get("probability", 0.0)) / retained_probability_mass
            for response in responses
        ]
        return probabilities, sum(probabilities), True

    unnormalized = [float(response.get("probability", 0.0)) for response in responses]
    total = sum(unnormalized)
    if total <= 0.0:
        raise ValueError("response probabilities have zero total mass")
    return [value / total for value in unnormalized], total, True


def weighted_mean_and_covariance(
    probabilities: list[float], observations: list[list[float]]
) -> tuple[list[float], list[list[float]]]:
    if len(probabilities) != len(observations):
        raise ValueError("probability and observation counts differ")
    if not observations:
        raise ValueError("at least one observation is required")
    total = sum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=PROBABILITY_TOLERANCE, abs_tol=PROBABILITY_TOLERANCE):
        probabilities = [value / total for value in probabilities]
    dimension = len(observations[0])
    mean = [
        sum(prob * obs[index] for prob, obs in zip(probabilities, observations))
        for index in range(dimension)
    ]
    covariance = [
        [
            sum(
                prob * (obs[row] - mean[row]) * (obs[col] - mean[col])
                for prob, obs in zip(probabilities, observations)
            )
            for col in range(dimension)
        ]
        for row in range(dimension)
    ]
    return mean, covariance


def _quadratic_form(weights: list[float], matrix: list[list[float]]) -> float:
    return sum(
        weights[row] * matrix[row][col] * weights[col]
        for row in range(len(weights))
        for col in range(len(weights))
    )


def _weighted_variance(probabilities: list[float], values: list[float]) -> tuple[float, float]:
    mean = sum(prob * value for prob, value in zip(probabilities, values))
    variance = sum(prob * (value - mean) ** 2 for prob, value in zip(probabilities, values))
    return mean, variance


def _covariance_columns(covariance: list[list[float]]) -> dict[str, float]:
    columns: dict[str, float] = {}
    for row, left in enumerate(COMPONENT_ORDER):
        for col, right in enumerate(COMPONENT_ORDER):
            if col <= row:
                continue
            columns[f"cov_{left}_{right}"] = covariance[row][col]
    return columns


def _correlation_matrix(covariance: list[list[float]]) -> list[list[float | None]]:
    correlations: list[list[float | None]] = []
    for row in range(len(covariance)):
        values: list[float | None] = []
        for col in range(len(covariance)):
            scale = covariance[row][row] * covariance[col][col]
            values.append(None if scale <= 0.0 else covariance[row][col] / math.sqrt(scale))
        correlations.append(values)
    return correlations


def candidate_covariance_record(
    *,
    game: str,
    ply_entry: dict[str, Any],
    move: dict[str, Any],
    game_weights: dict[str, float],
    selected_actual_move: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    responses = list(move.get("responses") or [])
    if not responses:
        return None, {"reason": "no saved responses"}

    missing = [
        response.get("uci", f"response_{index}")
        for index, response in enumerate(responses)
        if response.get("static_components") is None
    ]
    if missing:
        return None, {
            "reason": "static_components unavailable",
            "responses": missing,
            "candidate_uci": move.get("uci", ""),
        }

    retained_mass = _as_float(move.get("retained_probability_mass"))
    probabilities, probability_sum, normalized = normalize_probabilities(
        responses,
        retained_probability_mass=retained_mass,
    )
    observations = [
        [float(response["static_components"][name]) for name in COMPONENT_ORDER]
        for response in responses
    ]
    mean, covariance = weighted_mean_and_covariance(probabilities, observations)
    alpha = [float(game_weights.get(name, 1.0)) for name in COMPONENT_ORDER]
    variance_from_covariance = _quadratic_form(alpha, covariance)

    terminal_flags = [bool(response.get("static_terminal", False)) for response in responses]
    static_values = [_as_float(response.get("static_value")) for response in responses]
    if any(terminal_flags) or any(value is None for value in static_values):
        mean_static = None
        variance_direct = None
        consistency_error = None
    else:
        mean_static, variance_direct = _weighted_variance(
            probabilities, [float(value) for value in static_values]
        )
        consistency_error = variance_from_covariance - variance_direct

    row = {
        "game": game,
        "ply": int(ply_entry.get("ply", ply_entry.get("row", {}).get("ply", 0))),
        "move_number": ply_entry.get("row", {}).get("move_number"),
        "side": ply_entry.get("row", {}).get("side", ""),
        "player": ply_entry.get("row", {}).get("player", ""),
        "candidate_san": move.get("san", ""),
        "candidate_uci": move.get("uci", ""),
        "candidate_probability": move.get("probability"),
        "selected_actual_move": selected_actual_move,
        "candidate_was_actual_move": move.get("uci") == selected_actual_move,
        "number_of_responses": len(responses),
        "response_probability_sum": probability_sum,
        "response_probabilities_normalized": normalized,
        "mean_static_evaluation": mean_static,
        "variance_static_evaluation": variance_direct,
        "variance_static_from_covariance": variance_from_covariance,
        "variance_static_direct": variance_direct,
        "variance_consistency_error": consistency_error,
    }
    row.update({f"mean_{name}": mean[index] for index, name in enumerate(COMPONENT_ORDER)})
    row.update({f"var_{name}": covariance[index][index] for index, name in enumerate(COMPONENT_ORDER)})
    row.update(_covariance_columns(covariance))

    detail = {
        **row,
        "component_order": list(COMPONENT_ORDER),
        "mean_components": dict(zip(COMPONENT_ORDER, mean)),
        "covariance_matrix": covariance,
        "correlation_matrix": _correlation_matrix(covariance),
        "response_probabilities": probabilities,
    }
    return row, detail


def analyze_game(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    game_weights = {
        name: float(value)
        for name, value in dict(data.get("evaluation_weights", {})).items()
    }
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    warnings: list[str] = []
    for entry in data.get("plies", []):
        search = entry.get("search_result") or entry.get("actual_search_result") or {}
        weights = dict(game_weights)
        weights.update(dict(search.get("evaluation_weights") or {}))
        selected_actual_move = str(entry.get("row", {}).get("uci", ""))
        for move in search.get("moves", []):
            if not (move.get("selected_for_refinement") or move.get("actually_deepened")):
                continue
            row, detail_or_warning = candidate_covariance_record(
                game=path.stem,
                ply_entry=entry,
                move=move,
                game_weights=weights,
                selected_actual_move=selected_actual_move,
            )
            if row is None:
                reason = detail_or_warning.get("reason", "unavailable") if detail_or_warning else "unavailable"
                warnings.append(
                    f"{path}: ply {entry.get('ply')} candidate {move.get('uci', '')}: {reason}"
                )
                continue
            rows.append(row)
            if detail_or_warning is not None:
                details.append(detail_or_warning)
    return rows, details, warnings


def _fieldnames() -> list[str]:
    names = [
        "game",
        "ply",
        "move_number",
        "side",
        "player",
        "candidate_san",
        "candidate_uci",
        "candidate_probability",
        "selected_actual_move",
        "candidate_was_actual_move",
        "number_of_responses",
        "response_probability_sum",
        "response_probabilities_normalized",
        "mean_static_evaluation",
        "variance_static_evaluation",
    ]
    names.extend(f"mean_{name}" for name in COMPONENT_ORDER)
    names.extend(f"var_{name}" for name in COMPONENT_ORDER)
    names.extend(
        f"cov_{left}_{right}"
        for row, left in enumerate(COMPONENT_ORDER)
        for col, right in enumerate(COMPONENT_ORDER)
        if col > row
    )
    names.extend(
        [
            "variance_static_from_covariance",
            "variance_static_direct",
            "variance_consistency_error",
        ]
    )
    return names


def write_csv(rows: list[dict[str, Any]], output: Path | None) -> None:
    handle = output.open("w", newline="", encoding="utf-8") if output else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output:
            handle.close()


def _is_directory_output(path: Path, input_count: int) -> bool:
    return path.exists() and path.is_dir() or path.suffix == "" or input_count > 1


def _csv_output_path(base: Path | None, source: Path, input_count: int) -> Path:
    if base is None:
        return DEFAULT_COVARIANCE_DIR / f"{source.stem}.csv"
    if _is_directory_output(base, input_count):
        return base / f"{source.stem}.csv"
    return base


def _json_output_path(base: Path | None, source: Path, input_count: int) -> Path | None:
    if base is None:
        return None
    if _is_directory_output(base, input_count):
        return base / f"{source.stem}.json"
    return base


def print_ply_summary(details: list[dict[str, Any]], ply: int) -> None:
    selected = [detail for detail in details if int(detail["ply"]) == ply]
    selected.sort(key=lambda item: float(item.get("candidate_probability") or 0.0), reverse=True)
    if not selected:
        print(f"No analyzable deepened candidates found for ply {ply}.")
        return
    for detail in selected:
        print(
            f"Ply {ply} {detail['candidate_san']} ({detail['candidate_uci']}), "
            f"p={float(detail['candidate_probability']):.12g}, "
            f"responses={detail['number_of_responses']}"
        )
        print(
            "response probabilities: "
            + ", ".join(f"{value:.12g}" for value in detail["response_probabilities"])
        )
        print(
            "mean components: "
            + ", ".join(
                f"{name}={detail['mean_components'][name]:.12g}"
                for name in COMPONENT_ORDER
            )
        )
        print("covariance matrix:")
        for row in detail["covariance_matrix"]:
            print("  " + " ".join(f"{value: .12g}" for value in row))
        print(
            "variances: "
            + ", ".join(
                f"{name}={detail[f'var_{name}']:.12g}"
                for name in COMPONENT_ORDER
            )
        )
        print(
            f"mean E={detail['mean_static_evaluation']}, "
            f"var E={detail['variance_static_direct']}, "
            f"alpha^T Sigma alpha={detail['variance_static_from_covariance']}"
        )
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path, help="Game JSON file(s) or directories.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output file or directory. Defaults to data/analysis/covariance/<game>.csv.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional detailed JSON output file or directory.",
    )
    parser.add_argument("--ply", type=int, default=None, help="Print a readable summary for one ply.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    warnings: list[str] = []
    files = _game_files(args.inputs)
    wrote_paths: list[Path] = []
    for path in files:
        game_rows, game_details, game_warnings = analyze_game(path)
        rows.extend(game_rows)
        details.extend(game_details)
        warnings.extend(game_warnings)
        if game_rows:
            csv_path = _csv_output_path(args.output, path, len(files))
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_csv(game_rows, csv_path)
            wrote_paths.append(csv_path)
        json_path = _json_output_path(args.json_output, path, len(files))
        if json_path is not None and game_details:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(game_details, indent=2), encoding="utf-8")
            wrote_paths.append(json_path)

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if not rows:
        raise SystemExit(
            "No analyzable response records found. New JSON with response static_components is required."
        )

    for path in wrote_paths:
        print(f"Wrote {path}", file=sys.stderr)
    if args.ply is not None:
        print_ply_summary(details, args.ply)


if __name__ == "__main__":
    main()
