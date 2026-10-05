"""Match simulation and logging."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import chess
import chess.pgn

from .evaluation import EvaluationWeights, StaticEvaluator
from .features import PIECE_VALUES, material_balance
from .measure import KAPPA, Style
from .player import ThermoPlayer
from .search import LandscapeObservation, SearchMode, SearchResult, validate_depth
from .thermodynamics import decompose_transition



_CAPTURE_BASE_COUNTS = {
    chess.PAWN: 8,
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK: 2,
    chess.QUEEN: 1,
}


def captured_material(board: chess.Board) -> Dict[str, object]:
    def side_payload(captured_color: chess.Color) -> Dict[str, object]:
        pieces = {}
        value = 0.0
        for piece_type, initial in _CAPTURE_BASE_COUNTS.items():
            count = max(0, initial - len(board.pieces(piece_type, captured_color)))
            symbol = chess.piece_symbol(piece_type)
            pieces[symbol.upper()] = count
            value += count * PIECE_VALUES[piece_type]
        return {"pieces": pieces, "value": value}

    balance = material_balance(board)
    return {
        "white": side_payload(chess.BLACK),
        "black": side_payload(chess.WHITE),
        "balance": balance,
        "white_advantage": max(0.0, balance),
        "black_advantage": max(0.0, -balance),
    }

STRATEGY_NAMES = (
    "material_conservative",
    "pressure_aggressive",
    "tactical_attacker",
    "positional_controller",
)


@dataclass
class MatchConfig:
    max_plies: int = 120
    seed: int = 1
    beta_white: float = 4.0
    beta_black: float = 4.0
    kappa: float = KAPPA
    white_strategy: str = "material_conservative"
    black_strategy: str = "pressure_aggressive"
    depth: int = 3
    search_mode: SearchMode = "accurate"
    adaptive_c: float = 0.3
    games_dir: Path = Path("data/games")
    match_name: str | None = None
    stop_only_on_mate_or_stalemate: bool = False
    stop_on_threefold_repetition: bool = True
    viewer_workers: int = 1
    search_workers: int | None = 1
    parallel_min_branches: int = 8
    profile: bool = False

    def __post_init__(self) -> None:
        self.depth = validate_depth(self.depth)
        if self.kappa <= 0.0:
            raise ValueError("kappa must be greater than 0")
        if self.search_mode != "accurate":
            raise ValueError("search_mode must be 'accurate'")
        if self.adaptive_c <= 0.0:
            raise ValueError("adaptive_c must be greater than 0")
        if self.viewer_workers < 1:
            raise ValueError("viewer_workers must be at least 1")
        if self.search_workers is not None and self.search_workers < 1:
            raise ValueError("search_workers must be at least 1")
        if self.parallel_min_branches < 1:
            raise ValueError("parallel_min_branches must be at least 1")
        if self.white_strategy not in STRATEGY_NAMES:
            raise ValueError(f"unknown white strategy: {self.white_strategy}")
        if self.black_strategy not in STRATEGY_NAMES:
            raise ValueError(f"unknown black strategy: {self.black_strategy}")
        if self.match_name is None:
            self.match_name = default_match_name(self)


def conservative_material_style() -> Style:
    return Style(
        material=2.4,
        center=0.5,
        development=0.6,
        castling=0.9,
        king_safety=1.0,
        king_pressure=0.3,
    )


def pressure_aggressive_style() -> Style:
    return Style(
        material=0.9,
        center=1.1,
        development=0.6,
        castling=0.4,
        king_safety=0.4,
        king_pressure=1.8,
    )


def tactical_attacker_style() -> Style:
    return Style(
        material=1.0,
        center=1.4,
        development=0.5,
        castling=0.3,
        king_safety=0.5,
        king_pressure=2.2,
    )


def positional_controller_style() -> Style:
    return Style(
        material=1.2,
        center=1.8,
        development=0.9,
        castling=0.8,
        king_safety=1.1,
        king_pressure=0.6,
    )


def strategy_style(name: str) -> Style:
    builders = {
        "material_conservative": conservative_material_style,
        "pressure_aggressive": pressure_aggressive_style,
        "tactical_attacker": tactical_attacker_style,
        "positional_controller": positional_controller_style,
    }
    try:
        return builders[name]()
    except KeyError as exc:
        raise ValueError(f"unknown strategy: {name}") from exc


def _filename_number(value: float | int) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def default_match_name(config: MatchConfig) -> str:
    white = (
        f"{config.white_strategy}_b{_filename_number(config.beta_white)}"
    )
    black = (
        f"{config.black_strategy}_b{_filename_number(config.beta_black)}"
    )
    return f"{white}_vs_{black}_depth{config.depth}"



def _termination_reason(board: chess.Board, config: MatchConfig) -> str | None:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.is_seventyfive_moves():
        return "seventyfive_move_rule"
    if board.is_fivefold_repetition():
        return "fivefold_repetition"
    if config.stop_on_threefold_repetition and board.is_repetition(3):
        return "threefold_repetition"
    if not config.stop_only_on_mate_or_stalemate:
        if board.can_claim_fifty_moves():
            return "fifty_move_rule"
        if config.stop_on_threefold_repetition and board.can_claim_threefold_repetition():
            return "threefold_repetition_claim"
    return None


def _terminal_result(
    board: chess.Board, reached_ply_limit: bool, config: MatchConfig
) -> tuple[str, str]:
    reason = _termination_reason(board, config)
    if reason is not None:
        if reason == "checkmate":
            return board.result(claim_draw=False), reason
        return "1/2-1/2", reason
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


def _candidate_scores_from_result(result: SearchResult, maximize: bool) -> List[Dict[str, object]]:
    scores = [move.as_dict() for move in result.moves]
    fallback = float("-inf") if maximize else float("inf")
    return sorted(
        scores,
        key=lambda item: float(item["g_tilde"]) if item.get("g_tilde") is not None else fallback,
        reverse=maximize,
    )


def _viewer_panel_from_result(result: SearchResult, board_turn: chess.Color) -> Dict[str, object]:
    return result.as_viewer_panel(board_turn)


def _empty_counterfactual_panel(side: str) -> Dict[str, object]:
    return {
        "actual_decision": False,
        "counterfactual": True,
        "counterfactual_available": False,
        "side": side,
        "expected_value": None,
        "entropy": None,
        "effective_moves": None,
        "expanded_count": None,
        "moves": [],
        "sort_direction": "none",
    }

def _thermodynamic_transition_record(
    player: ThermoPlayer,
    old_result: SearchResult,
    new_ply: int,
    new_result: SearchResult,
    evaluator: StaticEvaluator,
) -> Dict[str, object]:
    search = player.search(evaluator)
    old_board = chess.Board(old_result.board_fen)
    new_board = chess.Board(new_result.board_fen)

    def shallow_observations(board: chess.Board):
        landscape = search.landscape(board)
        branches = tuple(
            LandscapeObservation(
                move=record.move,
                uci=record.uci,
                probability=record.probability,
                observable_value=record.static_after,
                was_deepened=False,
                depth_used=0,
                search_mode=player.search_mode,
                depth=0,
            )
            for record in landscape.records
        )
        return landscape, branches

    old_landscape, old_branches = shallow_observations(old_board)
    new_landscape, new_branches = shallow_observations(new_board)
    decomposition = decompose_transition(
        old_branches,
        new_branches,
    )
    return {
        "decomposition_type": "realized_full_shallow_same_player_transition",
        "player": player.name,
        "side": _side_name(player.color),
        "old_ply": old_result.diagnostics.get("ply", None),
        "new_ply": new_ply,
        "fen_old": old_result.board_fen,
        "fen_new": new_result.board_fen,
        "U_old": decomposition.u_before,
        "U_new": decomposition.u_after,
        "realized_delta_u": decomposition.delta_u,
        "realized_delta_q": decomposition.delta_q,
        "realized_delta_w": decomposition.delta_w,
        "realized_delta_a": decomposition.delta_a,
        "N_eff_old": old_landscape.effective_moves,
        "N_eff_new": new_landscape.effective_moves,
        "depth": player.depth,
        "search_mode": player.search_mode,
        "old_support_size": decomposition.old_support_size,
        "new_support_size": decomposition.new_support_size,
        "common_support_size": decomposition.common_support_size,
        "common_mass_old": decomposition.common_mass_old,
        "common_mass_new": decomposition.common_mass_new,
        "old_only_mass": decomposition.old_only_probability_mass,
        "new_only_mass": decomposition.new_only_probability_mass,
        "decomposition_error": decomposition.decomposition_error,
    }


def _realized_cycle_payload(
    transition: Dict[str, object] | None,
    next_side: str,
    next_san: str,
    next_uci: str,
) -> Dict[str, object] | None:
    """Viewer-ready completed same-player cycle.

    The first two one-ply deltas are the realized static board-value changes
    stored on the transition: E(B_m)-E(B) and E(B_mr)-E(B_m). The Q/W/A
    terms are the realized full shallow same-player transition for ``next_side``.
    """

    if transition is None:
        return None
    side = str(transition.get("side", ""))
    if side != next_side:
        return None
    action_side = str(transition.get("realized_action_side", ""))
    response_side = str(transition.get("realized_response_side", ""))
    if not action_side or not response_side or not next_san:
        return None
    return {
        "side": side,
        "label": f"{side.capitalize()} cycle",
        "moves": [
            {
                "san": transition.get("realized_action_san", ""),
                "uci": transition.get("realized_action_uci", ""),
                "color": action_side,
            },
            {
                "san": transition.get("realized_response_san", ""),
                "uci": transition.get("realized_response_uci", ""),
                "color": response_side,
            },
            {"san": next_san, "uci": next_uci, "color": next_side},
        ],
        "first_side": action_side,
        "second_side": response_side,
        "delta_u_first": transition.get("realized_action_delta_u"),
        "delta_u_second": transition.get("realized_response_delta_u"),
        "delta_u_by_color": {
            action_side: transition.get("realized_action_delta_u"),
            response_side: transition.get("realized_response_delta_u"),
        },
        "delta_u_cycle": transition.get("realized_delta_u"),
        "delta_q_cycle": transition.get("realized_delta_q"),
        "delta_w_cycle": transition.get("realized_delta_w"),
        "delta_a_cycle": transition.get("realized_delta_a"),
        "decomposition_error": transition.get("decomposition_error"),
        "old_ply": transition.get("old_ply"),
        "new_ply": transition.get("new_ply"),
    }

def _viewer_state(
    board: chess.Board,
    index: int,
    move_label: str,
    san: str,
    uci: str,
    previous_row: Dict[str, object] | None,
    evaluator: StaticEvaluator,
    actual_result: SearchResult | None = None,
) -> Dict[str, object]:
    is_game_over = board.is_game_over(claim_draw=False)
    white_panel = _empty_counterfactual_panel("white")
    black_panel = _empty_counterfactual_panel("black")
    order: List[str] = []
    if actual_result is not None:
        panel = _viewer_panel_from_result(actual_result, board.turn)
        order = [move["uci"] for move in panel["moves"]]
        if actual_result.side == "white":
            white_panel = panel
        else:
            black_panel = panel
    return {
        "index": index,
        "ply": index,
        "move_label": move_label,
        "san": san,
        "uci": uci,
        "fen": board.fen(),
        "turn": _side_name(board.turn),
        "static_evaluation": evaluator.evaluate(board),
        "captured_material": captured_material(board),
        "prediction_error": previous_row.get("prediction_error") if previous_row else None,
        "is_game_over": is_game_over,
        "result": board.result(claim_draw=False) if is_game_over else "*",
        "actual_search_result": actual_result.as_dict() if actual_result is not None else None,
        "actual_side": actual_result.side if actual_result is not None else None,
        "white_realized_cycle": None,
        "black_realized_cycle": None,
        "white_panel": white_panel,
        "black_panel": black_panel,
        "order": order,
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
    resolved_white_style = white_style or strategy_style(config.white_strategy)
    resolved_black_style = black_style or strategy_style(config.black_strategy)
    white = ThermoPlayer(
        name="White custom" if white_style is not None else f"White {config.white_strategy.replace('_', ' ')}",
        color=chess.WHITE,
        style=resolved_white_style,
        beta=config.beta_white,
        depth=config.depth,
        adaptive_c=config.adaptive_c,
        search_mode=config.search_mode,
        search_workers=config.search_workers,
        parallel_min_branches=config.parallel_min_branches,
        kappa=config.kappa,
    )
    black = ThermoPlayer(
        name="Black custom" if black_style is not None else f"Black {config.black_strategy.replace('_', ' ')}",
        color=chess.BLACK,
        style=resolved_black_style,
        beta=config.beta_black,
        depth=config.depth,
        adaptive_c=config.adaptive_c,
        search_mode=config.search_mode,
        search_workers=config.search_workers,
        parallel_min_branches=config.parallel_min_branches,
        kappa=config.kappa,
    )
    players = {chess.WHITE: white, chess.BLACK: black}

    csv_dir = config.games_dir / "csv"
    json_dir = config.games_dir / "json"
    pgn_dir = config.games_dir / "pgn"
    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    pgn_dir.mkdir(parents=True, exist_ok=True)

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Thermodynamic toy chess simulation"
    game.headers["White"] = white.name
    game.headers["Black"] = black.name
    node = game
    rows: List[Dict[str, object]] = []
    nested: List[Dict[str, object]] = []
    thermodynamic_transitions: List[Dict[str, object]] = []
    last_thermo_state: Dict[chess.Color, Dict[str, object]] = {}
    viewer_states: List[Dict[str, object]] = [
        _viewer_state(board, 0, "Start", "", "", None, evaluator)
    ]

    reached_ply_limit = True
    for ply in range(1, config.max_plies + 1):
        if _termination_reason(board, config) is not None:
            reached_ply_limit = False
            break

        player = players[board.turn]
        fen_before = board.fen()
        static_before = evaluator.evaluate(board)
        current_analysis = player.analyze(board, evaluator)
        search_result = current_analysis.search_result
        result_diag = {**search_result.diagnostics, "ply": ply}
        search_result = SearchResult(
            board_fen=search_result.board_fen,
            player=search_result.player,
            side=search_result.side,
            depth=search_result.depth,
            search_mode=search_result.search_mode,
            beta=search_result.beta,
            adaptive_c=search_result.adaptive_c,
            kappa=search_result.kappa,
            U=search_result.U,
            entropy=search_result.entropy,
            N_eff=search_result.N_eff,
            K=search_result.K,
            selected_uci=search_result.selected_uci,
            moves=search_result.moves,
            diagnostics=result_diag,
            evaluation_weights=search_result.evaluation_weights,
        )
        current_landscape = current_analysis.landscape
        thermo_transition = None
        previous_thermo = last_thermo_state.get(player.color)
        if previous_thermo is not None:
            thermo_transition = _thermodynamic_transition_record(
                player, previous_thermo, ply, search_result, evaluator
            )
            if len(rows) >= 2:
                action_row = rows[-2]
                response_row = rows[-1]
                if action_row.get("side") == _side_name(player.color):
                    action_delta = float(action_row["E_after"]) - float(action_row["E_before"])
                    response_delta = float(response_row["E_after"]) - float(response_row["E_before"])
                    thermo_transition.update({
                        "realized_action_side": action_row.get("side"),
                        "realized_action_san": action_row.get("san"),
                        "realized_action_uci": action_row.get("uci"),
                        "realized_response_side": response_row.get("side"),
                        "realized_response_san": response_row.get("san"),
                        "realized_response_uci": response_row.get("uci"),
                        "realized_action_delta_u": action_delta,
                        "realized_response_delta_u": response_delta,
                        "white_action_delta_u": action_delta if action_row.get("side") == "white" else response_delta if response_row.get("side") == "white" else None,
                        "black_response_delta_u": response_delta if response_row.get("side") == "black" else action_delta if action_row.get("side") == "black" else None,
                    })
            thermodynamic_transitions.append(thermo_transition)
        if viewer_states:
            viewer_states[-1] = _viewer_state(
                board,
                viewer_states[-1]["index"],
                viewer_states[-1]["move_label"],
                viewer_states[-1]["san"],
                viewer_states[-1]["uci"],
                rows[-1] if rows else None,
                evaluator,
                search_result,
            )
            viewer_states[-1]["thermodynamic_transition"] = thermo_transition
        last_thermo_state[player.color] = search_result
        choice = player.choose_from_result(search_result)
        candidate_scores = _candidate_scores_from_result(search_result, player.color == chess.WHITE)
        if choice.move is None:
            break
        if thermo_transition is not None and viewer_states:
            cycle_payload = _realized_cycle_payload(
                thermo_transition,
                _side_name(player.color),
                choice.san,
                choice.uci,
            )
            if cycle_payload is not None:
                viewer_states[-1][f"{_side_name(player.color)}_realized_cycle"] = cycle_payload

        board.push(choice.move)
        node = node.add_variation(choice.move)
        static_after = evaluator.evaluate(board)
        repetition_count = _repetition_count(board)

        cycle_delta_u = thermo_transition.get("realized_delta_u") if thermo_transition else None
        cycle_delta_q = thermo_transition.get("realized_delta_q") if thermo_transition else None
        cycle_delta_w = thermo_transition.get("realized_delta_w") if thermo_transition else None
        cycle_delta_a = thermo_transition.get("realized_delta_a") if thermo_transition else None
        cycle_error = thermo_transition.get("decomposition_error") if thermo_transition else None
        realized_action_delta_u = thermo_transition.get("realized_action_delta_u") if thermo_transition else None
        realized_response_delta_u = thermo_transition.get("realized_response_delta_u") if thermo_transition else None

        if _termination_reason(board, config) is not None:
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
            "depth": config.depth,
            "search_mode": config.search_mode,
            "adaptive_c": config.adaptive_c,
            "U_current": search_result.U,
            "terminal_u": choice.value,
            "g_tilde": choice.delta_u,
            "U_player_current": search_result.U,
            "U_before": thermo_transition.get("U_old") if thermo_transition else None,
            "U_after": thermo_transition.get("U_new") if thermo_transition else None,
            "realized_delta_u": cycle_delta_u,
            "realized_delta_q": cycle_delta_q,
            "realized_delta_w": cycle_delta_w,
            "realized_delta_a": cycle_delta_a,
            "realized_action_delta_u": realized_action_delta_u,
            "realized_response_delta_u": realized_response_delta_u,
            "decomposition_error": cycle_error,
            "entropy_current": search_result.entropy,
            "N_eff": search_result.N_eff,
            "N_eff_before": search_result.N_eff,
            "N_eff_after": thermo_transition.get("N_eff_new") if thermo_transition else None,
            "effective_moves_current": search_result.N_eff,
            "K_expanded": search_result.K,
            "old_support_size": thermo_transition.get("old_support_size") if thermo_transition else None,
            "new_support_size": thermo_transition.get("new_support_size") if thermo_transition else None,
            "common_support_size": thermo_transition.get("common_support_size") if thermo_transition else None,
            "common_mass_old": thermo_transition.get("common_mass_old") if thermo_transition else None,
            "common_mass_new": thermo_transition.get("common_mass_new") if thermo_transition else None,
            "old_only_probability_mass": thermo_transition.get("old_only_mass") if thermo_transition else None,
            "new_only_probability_mass": thermo_transition.get("new_only_mass") if thermo_transition else None,
            "nodes_evaluated": search_result.diagnostics["nodes_evaluated"],
            "static_evaluations": search_result.diagnostics["static_evaluations"],
            "recursive_nodes": search_result.diagnostics["recursive_nodes"],
            "maximum_depth_reached": search_result.diagnostics["maximum_depth_reached"],
            "average_K": search_result.diagnostics["average_k"],
            "cache_hits": search_result.diagnostics["cache_hits"],
            "search_elapsed_time": search_result.diagnostics["elapsed_time"],
            "selected_move": choice.uci,
            "selected_for_refinement": choice.uci in search_result.selected_uci,
            "selected_reply_expected_value": choice.value,
            "predicted_reply_expected_value": choice.value,
            "predicted_reply_entropy": None,
            "reply_realized_value": None,
            "prediction_error": None,
            "repetition_count": repetition_count,
            "is_twofold_repetition": board.is_repetition(2),
            "is_threefold_repetition": board.is_repetition(3),
            "can_claim_threefold_repetition": board.can_claim_threefold_repetition(),
            "is_fivefold_repetition": board.is_fivefold_repetition(),
            "is_seventyfive_moves": board.is_seventyfive_moves(),
            "can_claim_fifty_moves": board.can_claim_fifty_moves(),
            "is_insufficient_material": board.is_insufficient_material(),
            "is_checkmate": board.is_checkmate(),
            "is_stalemate": board.is_stalemate(),
        }
        rows.append(row)
        nested.append(
            {
                "ply": ply,
                "row": row,
                "current_landscape": {"fen": search_result.board_fen, "entropy": search_result.entropy, "effective_moves": search_result.N_eff, "expected_value": search_result.U, "moves": [move.as_dict() for move in search_result.moves]},
                "search_result": search_result.as_dict(),
                "candidate_move_scores": candidate_scores,
                "predicted_reply_landscape": None,
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
                evaluator,
            )
        )
        if _termination_reason(board, config) is not None:
            break

    csv_path = csv_dir / f"{config.match_name}.csv"
    json_path = json_dir / f"{config.match_name}.json"
    pgn_path = pgn_dir / f"{config.match_name}.pgn"

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    result, terminal_reason = _terminal_result(board, reached_ply_limit, config)

    payload = {
        "config": {
            **asdict(config),
            "games_dir": str(config.games_dir),
            "csv_dir": str(csv_dir),
            "json_dir": str(json_dir),
            "pgn_dir": str(pgn_dir),
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
        "final_is_seventyfive_moves": board.is_seventyfive_moves(),
        "final_can_claim_fifty_moves": board.can_claim_fifty_moves(),
        "final_is_insufficient_material": board.is_insufficient_material(),
        "viewer_states": viewer_states,
        "thermodynamic_transitions": thermodynamic_transitions,
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
