"""Match simulation and logging."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List

import chess
import chess.pgn

from .evaluation import EvaluationWeights, StaticEvaluator
from .measure import Style
from .player import ThermoPlayer
from .search import AdaptiveDepthThresholds
from .thermodynamics import decompose_transition


STRATEGY_NAMES = (
    "material_conservative",
    "activity_aggressive",
    "tactical_attacker",
    "positional_controller",
)


@dataclass
class MatchConfig:
    max_plies: int = 120
    seed: int = 1
    beta_white: float = 4.0
    beta_black: float = 4.0
    white_strategy: str = "material_conservative"
    black_strategy: str = "activity_aggressive"
    white_solidness: float | None = None
    black_solidness: float | None = None
    depth: int = 4
    adaptive_c: float = 0.3
    depth4_max_neff: float = 4.0
    depth3_max_neff: float = 8.0
    depth2_max_neff: float = 15.0
    results_dir: Path = Path("data/results")
    games_dir: Path = Path("data/games")
    match_name: str = "thermo_match"
    stop_only_on_mate_or_stalemate: bool = False
    stop_on_threefold_repetition: bool = True

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError("depth must be at least 0")
        if self.adaptive_c <= 0.0:
            raise ValueError("adaptive_c must be greater than 0")
        if self.white_strategy not in STRATEGY_NAMES:
            raise ValueError(f"unknown white strategy: {self.white_strategy}")
        if self.black_strategy not in STRATEGY_NAMES:
            raise ValueError(f"unknown black strategy: {self.black_strategy}")
        for name, value in (
            ("white_solidness", self.white_solidness),
            ("black_solidness", self.black_solidness),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        AdaptiveDepthThresholds(
            self.depth4_max_neff,
            self.depth3_max_neff,
            self.depth2_max_neff,
        )


def conservative_material_style() -> Style:
    return Style(
        material=1.5,
        preservation=2.4,
        king_restriction=0.2,
        king_pressure=0.1,
        check=0.12,
        mate=80.0,
        activity=0.2,
        king_safety=0.8,
        center=0.2,
        promotion=0.8,
        castle_preserve=0.8,
        castle_deny=0.3,
        castle=1.0,
        development=0.8,
        phase_castle=0.8,
        phase_attack=0.4,
        solidness=0.75,
    )


def aggressive_activity_style() -> Style:
    return Style(
        material=0.9,
        preservation=1.2,
        king_restriction=1.1,
        king_pressure=0.6,
        check=1.2,
        mate=80.0,
        activity=2.0,
        king_safety=0.35,
        center=1.2,
        promotion=0.8,
        castle_preserve=0.35,
        castle_deny=0.45,
        castle=0.55,
        development=0.5,
        phase_castle=0.4,
        phase_attack=1.1,
        solidness=0.35,
    )


def tactical_attacker_style() -> Style:
    return Style(
        material=1.0,
        preservation=0.3,
        king_restriction=1.4,
        king_pressure=1.5,
        check=2.0,
        mate=80.0,
        activity=1.4,
        king_safety=0.8,
        center=1.4,
        promotion=1.0,
        castle_preserve=0.25,
        castle_deny=0.7,
        castle=0.4,
        development=0.4,
        phase_castle=0.3,
        phase_attack=1.4,
        solidness=0.25,
    )


def positional_controller_style() -> Style:
    return Style(
        material=1.2,
        preservation=1.7,
        king_restriction=0.8,
        king_pressure=0.3,
        check=0.25,
        mate=80.0,
        activity=0.8,
        king_safety=1.1,
        center=1.8,
        promotion=0.8,
        castle_preserve=0.7,
        castle_deny=0.4,
        castle=0.9,
        development=0.9,
        phase_castle=0.9,
        phase_attack=0.6,
        solidness=0.8,
    )


def strategy_style(name: str) -> Style:
    builders = {
        "material_conservative": conservative_material_style,
        "activity_aggressive": aggressive_activity_style,
        "tactical_attacker": tactical_attacker_style,
        "positional_controller": positional_controller_style,
    }
    try:
        return builders[name]()
    except KeyError as exc:
        raise ValueError(f"unknown strategy: {name}") from exc



def _is_mate_or_stalemate(board: chess.Board) -> bool:
    return board.is_checkmate() or board.is_stalemate()


def _terminal_result(
    board: chess.Board, reached_ply_limit: bool, stop_on_threefold_repetition: bool
) -> tuple[str, str]:
    if board.is_checkmate():
        return board.result(claim_draw=False), "checkmate"
    if board.is_stalemate():
        return "1/2-1/2", "stalemate"
    if stop_on_threefold_repetition and board.is_repetition(3):
        return "1/2-1/2", "threefold_repetition"
    if reached_ply_limit:
        return "*", "max_plies"
    return "*", "not_terminal"


def _repetition_count(board: chess.Board, cap: int = 5) -> int:
    count = 1
    for candidate in range(2, cap + 1):
        if board.is_repetition(candidate):
            count = candidate
        else:
            break
    return count

def _side_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def _prediction_error_for_previous(
    rows: List[Dict[str, object]],
    board_after_actual_reply: chess.Board,
    evaluator: StaticEvaluator,
) -> None:
    if not rows:
        return
    previous = rows[-1]
    expected = previous.get("predicted_reply_expected_value")
    if expected is None:
        return
    realized = evaluator.evaluate(board_after_actual_reply)
    previous["reply_realized_value"] = realized
    previous["prediction_error"] = realized - float(expected)


def _candidate_scores_for_choice(
    player: ThermoPlayer, board: chess.Board, evaluator: StaticEvaluator
) -> List[Dict[str, object]]:
    analysis = player.analyze(board, evaluator)
    scores: List[Dict[str, object]] = []
    maximize = player.color == chess.WHITE
    for candidate in analysis.candidates:
        after = board.copy(stack=False)
        after.push(candidate.move)
        reply = player.landscape(after, evaluator)
        scores.append(
            {
                **candidate.as_dict(),
                "reply_entropy": reply.entropy,
                "reply_effective_moves": reply.effective_moves,
            }
        )
    return sorted(scores, key=lambda item: item["selection_value"], reverse=maximize)


def _viewer_panel(
    player: ThermoPlayer,
    board: chess.Board,
    evaluator: StaticEvaluator,
) -> Dict[str, object]:
    analysis = player.analyze(board, evaluator)
    landscape = analysis.landscape
    records = {record.uci: record for record in landscape.records}
    search = player.search(evaluator)
    old_branches = search.branch_observations(board)
    moves = []
    for candidate in analysis.candidates:
        record = records[candidate.uci]
        after = board.copy(stack=False)
        after.push(candidate.move)
        decomposition = decompose_transition(
            old_branches,
            search.branch_observations(after),
            u_before=analysis.current_value,
            u_after=candidate.value,
        )
        moves.append(
            {
                **record.as_dict(),
                "reply_expected_value": candidate.value,
                "delta_u": candidate.delta_u,
                "delta_q": decomposition.delta_q,
                "delta_w": decomposition.delta_w,
                "delta_a": decomposition.delta_a,
                "decomposition_error": decomposition.decomposition_error,
                "selected_for_deeper_analysis": candidate.selected_for_deeper_analysis,
            }
        )

    if board.turn == chess.WHITE:
        moves.sort(
            key=lambda item: (item["delta_u"], item["probability"], item["reply_expected_value"]),
            reverse=True,
        )
        sort_direction = "descending"
    else:
        moves.sort(
            key=lambda item: (item["delta_u"], -item["probability"], item["reply_expected_value"]),
        )
        sort_direction = "ascending"

    for rank, move in enumerate(moves, start=1):
        move["advantage_rank"] = rank

    return {
        "expected_value": analysis.current_value,
        "entropy": landscape.entropy,
        "effective_moves": landscape.effective_moves,
        "selected_depth": analysis.selected_depth,
        "expanded_count": analysis.expanded_count,
        "search_depth": player.depth,
        "adaptive_c": player.adaptive_c,
        "diagnostics": analysis.diagnostics,
        "sort_direction": sort_direction,
        "moves": moves,
    }


def _viewer_state(
    board: chess.Board,
    index: int,
    move_label: str,
    san: str,
    uci: str,
    previous_row: Dict[str, object] | None,
    white: ThermoPlayer,
    black: ThermoPlayer,
    evaluator: StaticEvaluator,
) -> Dict[str, object]:
    white_panel = _viewer_panel(white, board, evaluator)
    black_panel = _viewer_panel(black, board, evaluator)
    is_game_over = board.is_game_over(claim_draw=True)
    return {
        "index": index,
        "ply": index,
        "move_label": move_label,
        "san": san,
        "uci": uci,
        "fen": board.fen(),
        "turn": _side_name(board.turn),
        "static_evaluation": evaluator.evaluate(board),
        "prediction_error": previous_row.get("prediction_error") if previous_row else None,
        "is_game_over": is_game_over,
        "result": board.result(claim_draw=True) if is_game_over else "*",
        "white_panel": white_panel,
        "black_panel": black_panel,
        "order": [move["uci"] for move in white_panel["moves"]],
    }


def simulate_match(
    white_style: Style | None = None,
    black_style: Style | None = None,
    eval_weights: EvaluationWeights | None = None,
    config: MatchConfig | None = None,
) -> Dict[str, Path | List[Dict[str, object]]]:
    config = config or MatchConfig()
    random.seed(config.seed)
    evaluator = StaticEvaluator(eval_weights)
    depth_thresholds = AdaptiveDepthThresholds(
        config.depth4_max_neff,
        config.depth3_max_neff,
        config.depth2_max_neff,
    )
    resolved_white_style = white_style or strategy_style(config.white_strategy)
    resolved_black_style = black_style or strategy_style(config.black_strategy)
    if config.white_solidness is not None:
        resolved_white_style = replace(
            resolved_white_style, solidness=config.white_solidness
        )
    if config.black_solidness is not None:
        resolved_black_style = replace(
            resolved_black_style, solidness=config.black_solidness
        )
    white = ThermoPlayer(
        "White custom" if white_style is not None else f"White {config.white_strategy.replace('_', ' ')}",
        chess.WHITE,
        resolved_white_style,
        config.beta_white,
        config.depth,
        config.adaptive_c,
        depth_thresholds,
    )
    black = ThermoPlayer(
        "Black custom" if black_style is not None else f"Black {config.black_strategy.replace('_', ' ')}",
        chess.BLACK,
        resolved_black_style,
        config.beta_black,
        config.depth,
        config.adaptive_c,
        depth_thresholds,
    )
    players = {chess.WHITE: white, chess.BLACK: black}

    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.games_dir.mkdir(parents=True, exist_ok=True)

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Thermodynamic toy chess simulation"
    game.headers["White"] = white.name
    game.headers["Black"] = black.name
    node = game
    rows: List[Dict[str, object]] = []
    nested: List[Dict[str, object]] = []
    viewer_states: List[Dict[str, object]] = [
        _viewer_state(board, 0, "Start", "", "", None, white, black, evaluator)
    ]

    reached_ply_limit = True
    for ply in range(1, config.max_plies + 1):
        if _is_mate_or_stalemate(board) or (
            config.stop_on_threefold_repetition and board.is_repetition(3)
        ):
            reached_ply_limit = False
            break
        if not config.stop_only_on_mate_or_stalemate and not config.stop_on_threefold_repetition:
            if board.is_game_over(claim_draw=True):
                reached_ply_limit = False
                break

        player = players[board.turn]
        fen_before = board.fen()
        static_before = evaluator.evaluate(board)
        current_analysis = player.analyze(board, evaluator)
        current_landscape = current_analysis.landscape
        old_branches = player.search(evaluator).branch_observations(board)
        choice = player.choose(board, evaluator)
        candidate_scores = _candidate_scores_for_choice(player, board, evaluator)
        if choice.move is None:
            break

        board.push(choice.move)
        node = node.add_variation(choice.move)
        static_after = evaluator.evaluate(board)
        after_analysis = player.analyze(board, evaluator)
        new_branches = player.search(evaluator).branch_observations(board)
        decomposition = decompose_transition(
            old_branches,
            new_branches,
            u_before=current_analysis.current_value,
            u_after=after_analysis.current_value,
        )
        repetition_count = _repetition_count(board)

        if _is_mate_or_stalemate(board) or (
            config.stop_on_threefold_repetition and board.is_repetition(3)
        ):
            reached_ply_limit = False

        _prediction_error_for_previous(rows, board, evaluator)
        if rows and viewer_states:
            viewer_states[-1]["prediction_error"] = rows[-1].get("prediction_error")

        row = {
            "ply": ply,
            "move_number": board.fullmove_number if board.turn == chess.WHITE else board.fullmove_number,
            "side": _side_name(player.color),
            "player": player.name,
            "san": choice.san,
            "uci": choice.uci,
            "fen_before": fen_before,
            "fen_after": board.fen(),
            "E_before": static_before,
            "E_after": static_after,
            "search_depth": config.depth,
            "adaptive_c": config.adaptive_c,
            "U_current": current_analysis.current_value,
            "U_after_move": after_analysis.current_value,
            "delta_u": decomposition.delta_u,
            "U_player_current": current_analysis.current_value,
            "U_before": decomposition.u_before,
            "U_after": decomposition.u_after,
            "delta_U": decomposition.delta_u,
            "delta_Q": decomposition.delta_q,
            "delta_W": decomposition.delta_w,
            "delta_A": decomposition.delta_a,
            "decomposition_error": decomposition.decomposition_error,
            "entropy_current": current_landscape.entropy,
            "N_eff": current_analysis.effective_moves,
            "N_eff_before": current_analysis.effective_moves,
            "N_eff_after": after_analysis.effective_moves,
            "effective_moves_current": current_landscape.effective_moves,
            "selected_depth": current_analysis.selected_depth,
            "adaptive_depth_before": current_analysis.selected_depth,
            "adaptive_depth_after": after_analysis.selected_depth,
            "K_expanded": current_analysis.expanded_count,
            "old_support_size": decomposition.old_support_size,
            "new_support_size": decomposition.new_support_size,
            "common_support_size": decomposition.common_support_size,
            "common_mass_old": decomposition.common_mass_old,
            "common_mass_new": decomposition.common_mass_new,
            "common_probability_mass_old": decomposition.common_mass_old,
            "common_probability_mass_new": decomposition.common_mass_new,
            "old_only_probability_mass": decomposition.old_only_probability_mass,
            "new_only_probability_mass": decomposition.new_only_probability_mass,
            "boundary_accessibility": decomposition.boundary_accessibility,
            "nodes_evaluated": current_analysis.diagnostics["nodes_evaluated"],
            "static_evaluations": current_analysis.diagnostics["static_evaluations"],
            "recursive_nodes": current_analysis.diagnostics["recursive_nodes"],
            "maximum_depth_reached": current_analysis.diagnostics["maximum_depth_reached"],
            "average_K": current_analysis.diagnostics["average_k"],
            "cache_hits": current_analysis.diagnostics["cache_hits"],
            "search_elapsed_time": current_analysis.diagnostics["elapsed_time"],
            "selected_move": choice.uci,
            "selected_for_deeper_analysis": choice.uci
            in current_analysis.selected_uci,
            "selected_reply_expected_value": choice.value,
            "predicted_reply_expected_value": choice.value,
            "predicted_reply_entropy": choice.reply_landscape.entropy
            if choice.reply_landscape
            else None,
            "reply_realized_value": None,
            "prediction_error": None,
            "repetition_count": repetition_count,
            "is_twofold_repetition": board.is_repetition(2),
            "is_threefold_repetition": board.is_repetition(3),
            "can_claim_threefold_repetition": board.can_claim_threefold_repetition(),
            "is_fivefold_repetition": board.is_fivefold_repetition(),
            "is_checkmate": board.is_checkmate(),
            "is_stalemate": board.is_stalemate(),
        }
        rows.append(row)
        nested.append(
            {
                "ply": ply,
                "row": row,
                "current_landscape": current_landscape.as_dict(),
                "candidate_move_scores": candidate_scores,
                "adaptive_branches_before": [
                    branch.as_dict() for branch in old_branches
                ],
                "adaptive_branches_after": [
                    branch.as_dict() for branch in new_branches
                ],
                "predicted_reply_landscape": choice.reply_landscape.as_dict()
                if choice.reply_landscape
                else None,
            }
        )
        viewer_states.append(
            _viewer_state(
                board,
                ply,
                f"{ply}. {_side_name(player.color)} {choice.san}",
                choice.san,
                choice.uci,
                row,
                white,
                black,
                evaluator,
            )
        )
        if _is_mate_or_stalemate(board) or (
            config.stop_on_threefold_repetition and board.is_repetition(3)
        ):
            break

    csv_path = config.results_dir / f"{config.match_name}.csv"
    json_path = config.results_dir / f"{config.match_name}.json"
    pgn_path = config.games_dir / f"{config.match_name}.pgn"

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    result, terminal_reason = _terminal_result(
        board, reached_ply_limit, config.stop_on_threefold_repetition
    )

    payload = {
        "config": {
            **asdict(config),
            "results_dir": str(config.results_dir),
            "games_dir": str(config.games_dir),
        },
        "white_style": white.style.as_dict(),
        "black_style": black.style.as_dict(),
        "evaluation_weights": evaluator.weights.as_dict(),
        "result": result,
        "terminal_reason": terminal_reason,
        "final_fen": board.fen(),
        "final_repetition_count": _repetition_count(board),
        "final_can_claim_threefold_repetition": board.can_claim_threefold_repetition(),
        "final_is_fivefold_repetition": board.is_fivefold_repetition(),
        "viewer_states": viewer_states,
        "plies": nested,
    }
    game.headers["Result"] = payload["result"]
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with pgn_path.open("w", encoding="utf-8") as handle:
        print(game, file=handle)

    return {
        "rows": rows,
        "csv": csv_path,
        "json": json_path,
        "pgn": pgn_path,
    }
