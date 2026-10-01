"""Match simulation and logging."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List

import chess
import chess.pgn

from .evaluation import EvaluationWeights, StaticEvaluator
from .features import PIECE_VALUES, material_balance
from .measure import KAPPA, Style
from .player import ThermoPlayer
from .search import AdaptiveDepthThresholds, CandidateThermoMode, RefinementPolicy, SearchMode, SearchResult
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
    kappa: float = KAPPA
    white_strategy: str = "material_conservative"
    black_strategy: str = "activity_aggressive"
    white_solidness: float | None = None
    black_solidness: float | None = None
    cdepth: int = 1
    search_mode: SearchMode = "accurate"
    adaptive_c: float = 0.3
    depth: int | None = None
    depth4_max_neff: float = 4.0
    depth3_max_neff: float = 8.0
    depth2_max_neff: float = 15.0
    results_dir: Path = Path("data/results")
    games_dir: Path = Path("data/games")
    match_name: str | None = None
    stop_only_on_mate_or_stalemate: bool = False
    stop_on_threefold_repetition: bool = True
    viewer_workers: int = 1
    search_workers: int | None = 1
    parallel_min_branches: int = 8
    candidate_thermo_mode: CandidateThermoMode = "refined"
    refinement_policy: RefinementPolicy = "static_eval"
    profile: bool = False

    def __post_init__(self) -> None:
        if self.depth is not None:
            self.cdepth = self.depth
        self.depth = self.cdepth
        if self.kappa <= 0.0:
            raise ValueError("kappa must be greater than 0")
        if self.cdepth < 0:
            raise ValueError("cdepth must be at least 0")
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
        if self.candidate_thermo_mode not in {"selected", "refined", "all"}:
            raise ValueError("candidate_thermo_mode must be 'selected', 'refined', or 'all'")
        if self.refinement_policy not in {"static_eval", "probability"}:
            raise ValueError("refinement_policy must be 'static_eval' or 'probability'")
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
        if self.match_name is None:
            self.match_name = default_match_name(self)
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
        castle_preserve=2.8,
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
        castle_preserve=1.2,
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
        castle_preserve=1.0,
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
        castle_preserve=2.6,
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


def _filename_number(value: float | int) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def default_match_name(config: MatchConfig) -> str:
    white_solidness = (
        config.white_solidness
        if config.white_solidness is not None
        else strategy_style(config.white_strategy).solidness
    )
    black_solidness = (
        config.black_solidness
        if config.black_solidness is not None
        else strategy_style(config.black_strategy).solidness
    )
    white = (
        f"{config.white_strategy}_b{_filename_number(config.beta_white)}"
        f"_s{_filename_number(white_solidness)}"
    )
    black = (
        f"{config.black_strategy}_b{_filename_number(config.beta_black)}"
        f"_s{_filename_number(black_solidness)}"
    )
    return f"{white}_vs_{black}_cdepth{config.cdepth}"



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
    return sorted(scores, key=lambda item: item["selection_value"], reverse=maximize)


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
        "selected_depth": None,
        "expanded_count": None,
        "moves": [],
        "sort_direction": "none",
    }

def _thermodynamic_transition_record(
    player: ThermoPlayer,
    old_result: SearchResult,
    new_ply: int,
    new_result: SearchResult,
) -> Dict[str, object]:
    decomposition = decompose_transition(
        old_result.branch_observations(),
        new_result.branch_observations(),
        u_before=old_result.U,
        u_after=new_result.U,
    )
    return {
        "decomposition_type": "adaptive_same_player_transition",
        "player": player.name,
        "side": _side_name(player.color),
        "old_ply": old_result.diagnostics.get("ply", None),
        "new_ply": new_ply,
        "fen_old": old_result.board_fen,
        "fen_new": new_result.board_fen,
        "U_old": decomposition.u_before,
        "U_new": decomposition.u_after,
        "adaptive_same_player_delta_U": decomposition.delta_u,
        "adaptive_same_player_delta_Q": decomposition.delta_q,
        "adaptive_same_player_delta_W": decomposition.delta_w,
        "adaptive_same_player_delta_A": decomposition.delta_a,
        "delta_U_tilde": decomposition.delta_u,
        "delta_Q_tilde": decomposition.delta_q,
        "delta_W_tilde": decomposition.delta_w,
        "delta_A_tilde": decomposition.delta_a,
        "delta_U": decomposition.delta_u,
        "delta_Q": decomposition.delta_q,
        "delta_W": decomposition.delta_w,
        "delta_A": decomposition.delta_a,
        "N_eff_old": old_result.N_eff,
        "N_eff_new": new_result.N_eff,
        "cdepth": player.cdepth,
        "search_mode": player.search_mode,
        "K_old": old_result.K,
        "K_new": new_result.K,
        "adaptive_cdepth_old": old_result.selected_depth,
        "adaptive_cdepth_new": new_result.selected_depth,
        "adaptive_depth_old": old_result.selected_depth,
        "adaptive_depth_new": new_result.selected_depth,
        "old_support_size": decomposition.old_support_size,
        "new_support_size": decomposition.new_support_size,
        "common_support_size": decomposition.common_support_size,
        "common_mass_old": decomposition.common_mass_old,
        "common_mass_new": decomposition.common_mass_new,
        "old_only_mass": decomposition.old_only_probability_mass,
        "new_only_mass": decomposition.new_only_probability_mass,
        "decomposition_error": decomposition.decomposition_error,
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
        config.cdepth,
        config.adaptive_c,
        depth_thresholds,
        config.search_mode,
        config.search_workers,
        config.parallel_min_branches,
        config.candidate_thermo_mode,
        config.refinement_policy,
        kappa=config.kappa,
    )
    black = ThermoPlayer(
        "Black custom" if black_style is not None else f"Black {config.black_strategy.replace('_', ' ')}",
        chess.BLACK,
        resolved_black_style,
        config.beta_black,
        config.cdepth,
        config.adaptive_c,
        depth_thresholds,
        config.search_mode,
        config.search_workers,
        config.parallel_min_branches,
        config.candidate_thermo_mode,
        config.refinement_policy,
        kappa=config.kappa,
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
            cdepth=search_result.cdepth,
            search_mode=search_result.search_mode,
            beta=search_result.beta,
            adaptive_c=search_result.adaptive_c,
            refinement_policy=search_result.refinement_policy,
            U=search_result.U,
            entropy=search_result.entropy,
            N_eff=search_result.N_eff,
            K=search_result.K,
            selected_depth=search_result.selected_depth,
            selected_uci=search_result.selected_uci,
            moves=search_result.moves,
            diagnostics=result_diag,
        )
        current_landscape = current_analysis.landscape
        thermo_transition = None
        previous_thermo = last_thermo_state.get(player.color)
        if previous_thermo is not None:
            thermo_transition = _thermodynamic_transition_record(player, previous_thermo, ply, search_result)
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

        board.push(choice.move)
        node = node.add_variation(choice.move)
        static_after = evaluator.evaluate(board)
        repetition_count = _repetition_count(board)

        cycle_delta_u = thermo_transition.get("adaptive_same_player_delta_U") if thermo_transition else None
        cycle_delta_q = thermo_transition.get("adaptive_same_player_delta_Q") if thermo_transition else None
        cycle_delta_w = thermo_transition.get("adaptive_same_player_delta_W") if thermo_transition else None
        cycle_delta_a = thermo_transition.get("adaptive_same_player_delta_A") if thermo_transition else None
        cycle_error = thermo_transition.get("decomposition_error") if thermo_transition else None
        cycle_old_branches = previous_thermo.branch_observations() if previous_thermo else ()
        cycle_new_branches = search_result.branch_observations() if previous_thermo else ()

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
            "cdepth": config.cdepth,
            "search_depth": config.cdepth,
            "search_mode": config.search_mode,
            "adaptive_c": config.adaptive_c,
            "refinement_policy": config.refinement_policy,
            "U_current": search_result.U,
            "U_after_move": choice.value,
            "expected_delta_u_star": choice.delta_u,
            "candidate_delta_u": choice.delta_u,
            "delta_u": choice.delta_u,
            "delta_u_star": choice.delta_u,
            "U_player_current": search_result.U,
            "U_before": thermo_transition.get("U_old") if thermo_transition else None,
            "U_after": thermo_transition.get("U_new") if thermo_transition else None,
            "adaptive_same_player_delta_U": cycle_delta_u,
            "adaptive_same_player_delta_Q": cycle_delta_q,
            "adaptive_same_player_delta_W": cycle_delta_w,
            "adaptive_same_player_delta_A": cycle_delta_a,
            "cycle_delta_U": cycle_delta_u,
            "delta_U_tilde": cycle_delta_u,
            "delta_Q_tilde": cycle_delta_q,
            "delta_W_tilde": cycle_delta_w,
            "delta_A_tilde": cycle_delta_a,
            "delta_U": cycle_delta_u,
            "delta_Q": cycle_delta_q,
            "delta_W": cycle_delta_w,
            "delta_A": cycle_delta_a,
            "decomposition_error": cycle_error,
            "entropy_current": search_result.entropy,
            "N_eff": search_result.N_eff,
            "N_eff_before": search_result.N_eff,
            "N_eff_after": thermo_transition.get("N_eff_new") if thermo_transition else None,
            "effective_moves_current": search_result.N_eff,
            "selected_depth": search_result.selected_depth,
            "adaptive_cdepth": search_result.selected_depth,
            "adaptive_depth_before": search_result.selected_depth,
            "adaptive_depth_after": None,
            "K_expanded": search_result.K,
            "K_old": thermo_transition.get("K_old") if thermo_transition else None,
            "K_new": search_result.K if thermo_transition else None,
            "old_support_size": thermo_transition.get("old_support_size") if thermo_transition else None,
            "new_support_size": thermo_transition.get("new_support_size") if thermo_transition else None,
            "common_support_size": thermo_transition.get("common_support_size") if thermo_transition else None,
            "common_mass_old": thermo_transition.get("common_mass_old") if thermo_transition else None,
            "common_mass_new": thermo_transition.get("common_mass_new") if thermo_transition else None,
            "common_probability_mass_old": thermo_transition.get("common_mass_old") if thermo_transition else None,
            "common_probability_mass_new": thermo_transition.get("common_mass_new") if thermo_transition else None,
            "old_only_probability_mass": thermo_transition.get("old_only_mass") if thermo_transition else None,
            "new_only_probability_mass": thermo_transition.get("new_only_mass") if thermo_transition else None,
            "boundary_accessibility": None,
            "nodes_evaluated": search_result.diagnostics["nodes_evaluated"],
            "static_evaluations": search_result.diagnostics["static_evaluations"],
            "recursive_nodes": search_result.diagnostics["recursive_nodes"],
            "maximum_depth_reached": search_result.diagnostics["maximum_depth_reached"],
            "average_K": search_result.diagnostics["average_k"],
            "cache_hits": search_result.diagnostics["cache_hits"],
            "search_elapsed_time": search_result.diagnostics["elapsed_time"],
            "selected_move": choice.uci,
            "selected_for_deeper_analysis": choice.uci
            in search_result.selected_uci,
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
                "adaptive_branches_before": [
                    branch.as_dict() for branch in cycle_old_branches
                ],
                "adaptive_branches_after": [
                    branch.as_dict() for branch in cycle_new_branches
                ],
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

    csv_path = config.results_dir / f"{config.match_name}.csv"
    json_path = config.results_dir / f"{config.match_name}.json"
    pgn_path = config.games_dir / f"{config.match_name}.pgn"

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
