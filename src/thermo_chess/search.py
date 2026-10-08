"""Recursive subjective minimax search with entropy-adaptive breadth."""

from __future__ import annotations

import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Literal, Tuple

import chess

from .evaluation import StaticEvaluator
from .features import BoardFeatureContext, board_context
from .measure import KAPPA, MoveLandscape, MoveRecord, Style, deepening_count_for_quantile, move_distribution


PositionKey = Tuple[str, bool, bool, bool]
SearchMode = Literal["accurate"]
FULL_DEPTH_MAX_CONSIDERATION = 5
TWO_CYCLE_MAX_CONSIDERATION = 15
CacheKey = Tuple[PositionKey, int, int]


def validate_cdepth(cdepth: int) -> int:
    if not isinstance(cdepth, int) or isinstance(cdepth, bool):
        raise ValueError(
            "cdepth must be a positive integer representing the number "
            "of complete move-response cycles"
        )
    if cdepth < 1:
        raise ValueError(
            "cdepth must be a positive integer representing the number "
            "of complete move-response cycles"
        )
    return cdepth


def requested_recursive_plies_for_cdepth(cdepth: int) -> int:
    return 2 * validate_cdepth(cdepth) - 1


def adaptive_cycle_limit(deepening_branch_count: int, cdepth: int) -> int | None:
    if not isinstance(deepening_branch_count, int) or isinstance(deepening_branch_count, bool):
        raise ValueError("deepening_branch_count must be a non-negative integer")
    if deepening_branch_count < 0:
        raise ValueError("deepening_branch_count must be a non-negative integer")
    cdepth = validate_cdepth(cdepth)
    if deepening_branch_count == 0:
        return 0
    if deepening_branch_count == 1:
        return None
    if deepening_branch_count > TWO_CYCLE_MAX_CONSIDERATION:
        return min(1, cdepth)
    if deepening_branch_count >= FULL_DEPTH_MAX_CONSIDERATION + 1:
        return min(2, cdepth)
    return cdepth


def adaptive_recursive_plies_for_root(cycle_limit: int) -> int:
    if cycle_limit < 0:
        raise ValueError("cycle_limit must be non-negative")
    return 0 if cycle_limit == 0 else 2 * cycle_limit - 1


def adaptive_recursive_plies_from_node(cycle_limit: int, level: int) -> int:
    if cycle_limit < 0:
        raise ValueError("cycle_limit must be non-negative")
    if level < 0:
        raise ValueError("level must be non-negative")
    if cycle_limit == 0:
        return 0
    return 2 * cycle_limit - 1 if level % 2 == 1 else 2 * cycle_limit


def validate_deepening_quantile(deepening_quantile: float) -> float:
    if deepening_quantile <= 0.0 or deepening_quantile > 1.0:
        raise ValueError("deepening_quantile must satisfy 0 < q <= 1")
    return deepening_quantile


def adaptive_breadth(effective_moves: float, adaptive_c: float, legal_moves: int) -> int:
    """Deprecated compatibility helper for the former c*N_eff rule."""
    if adaptive_c <= 0.0:
        raise ValueError("adaptive_c must be greater than 0")
    if legal_moves <= 0:
        return 0
    return min(legal_moves, max(1, math.ceil(adaptive_c * effective_moves)))


def ranked_for_refinement(
    records: Iterable[MoveRecord],
    side_to_move: chess.Color | None = None,
) -> Tuple[MoveRecord, ...]:
    """Rank considered branches by renormalized transition probability."""
    return tuple(sorted(records, key=lambda record: record.probability, reverse=True))


def _validate_search_mode(search_mode: str) -> SearchMode:
    if search_mode != "accurate":
        raise ValueError("search_mode must be 'accurate'")
    return search_mode  # type: ignore[return-value]


def _move_from_uci(uci: str) -> chess.Move:
    return chess.Move.null() if uci == "__terminal__" else chess.Move.from_uci(uci)


@dataclass(frozen=True)
class ResponseObservation:
    move: chess.Move
    uci: str
    san: str
    probability: float
    response_value: float
    selected_for_refinement: bool
    depth_used: int
    static_value: float | None = None
    refinement_rank: int | None = None
    used_shallow_landscape: bool = False
    conditional_probability: float | None = None
    branch_g_tilde: float | None = None
    branch_q_tilde: float | None = None
    branch_w_tilde: float | None = None
    branch_a_tilde: float | None = None
    branch_decomposition_error: float | None = None
    static_components: Dict[str, float] | None = None
    board_fen: str | None = None
    static_terminal: bool = False
    static_terminal_reason: str | None = None
    selected_by_recursive_policy: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "uci": self.uci,
            "san": self.san,
            "probability": self.probability,
            "response_value": self.response_value,
            "value": self.response_value,
            "selected_for_refinement": self.selected_for_refinement,
            "recursive_plies_used": self.depth_used,
            "recursively_deepened": self.depth_used > 0,
            "static_value": self.static_value,
            "refinement_rank": self.refinement_rank,
            "used_shallow_landscape": self.used_shallow_landscape,
            "conditional_probability": self.conditional_probability,
            "branch_g_tilde": self.branch_g_tilde,
            "branch_q_tilde": self.branch_q_tilde,
            "branch_w_tilde": self.branch_w_tilde,
            "branch_a_tilde": self.branch_a_tilde,
            "branch_decomposition_error": self.branch_decomposition_error,
            "static_components": self.static_components,
            "board_fen": self.board_fen,
            "static_terminal": self.static_terminal,
            "static_terminal_reason": self.static_terminal_reason,
            "selected_by_recursive_policy": self.selected_by_recursive_policy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ResponseObservation":
        uci = str(data["uci"])
        return cls(
            move=_move_from_uci(uci),
            uci=uci,
            san=str(data.get("san", uci)),
            probability=float(data["probability"]),
            response_value=float(data.get("response_value", data.get("value", 0.0))),
            selected_for_refinement=bool(data.get("selected_for_refinement", False)),
            depth_used=int(data.get("recursive_plies_used", data.get("depth_used", 0))),
            static_value=None if data.get("static_value") is None else float(data["static_value"]),
            refinement_rank=None if data.get("refinement_rank") is None else int(data["refinement_rank"]),
            used_shallow_landscape=bool(data.get("used_shallow_landscape", False)),
            conditional_probability=None if data.get("conditional_probability", data.get("conditional_probability")) is None else float(data.get("conditional_probability", data.get("conditional_probability"))),
            branch_g_tilde=None if data.get("branch_g_tilde") is None else float(data["branch_g_tilde"]),
            branch_q_tilde=None if data.get("branch_q_tilde") is None else float(data["branch_q_tilde"]),
            branch_w_tilde=None if data.get("branch_w_tilde") is None else float(data["branch_w_tilde"]),
            branch_a_tilde=None if data.get("branch_a_tilde") is None else float(data["branch_a_tilde"]),
            branch_decomposition_error=None if data.get("branch_decomposition_error") is None else float(data["branch_decomposition_error"]),
            static_components=None if data.get("static_components") is None else {str(key): float(value) for key, value in dict(data["static_components"]).items()},
            board_fen=None if data.get("board_fen") is None else str(data["board_fen"]),
            static_terminal=bool(data.get("static_terminal", False)),
            static_terminal_reason=None if data.get("static_terminal_reason") is None else str(data["static_terminal_reason"]),
            selected_by_recursive_policy=bool(data.get("selected_by_recursive_policy", False)),
        )


@dataclass(frozen=True)
class LandscapeObservation:
    move: chess.Move
    uci: str
    probability: float
    observable_value: float
    was_deepened: bool
    depth_used: int
    response_neff: float | None = None
    response_k: int | None = None
    total_probability_mass: float | None = None
    deepened_probability_mass: float | None = None
    nondeepened_probability_mass: float | None = None
    response_branches: Tuple[ResponseObservation, ...] = ()
    search_mode: SearchMode = "accurate"
    remaining_plies: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "uci": self.uci,
            "probability": self.probability,
            "observable_value": self.observable_value,
            "branch_value": self.observable_value,
            "was_deepened": self.was_deepened,
            "selected_for_refinement": self.was_deepened,
            "actually_deepened": self.depth_used > 0,
            "recursive_plies_used": self.depth_used,
            "remaining_plies": self.remaining_plies,
            "search_mode": self.search_mode,
            "response_neff": self.response_neff,
            "response_k": self.response_k,
            "total_probability_mass": self.total_probability_mass,
            "deepened_probability_mass": self.deepened_probability_mass,
            "nondeepened_probability_mass": self.nondeepened_probability_mass,
            "response_branches": [branch.as_dict() for branch in self.response_branches],
        }


@dataclass(frozen=True)
class ResponseEvaluation:
    move: chess.Move
    uci: str
    san: str
    probability: float
    value: float
    selected_for_refinement: bool
    static_value: float | None = None
    refinement_rank: int | None = None
    depth_used: int = 0
    terminal_u: float | None = None
    branch_g_tilde: float | None = None
    branch_q_tilde: float | None = None
    branch_w_tilde: float | None = None
    branch_a_tilde: float | None = None
    branch_decomposition_error: float | None = None
    conditional_probability: float | None = None
    used_shallow_landscape: bool = False
    static_components: Dict[str, float] | None = None
    board_fen: str | None = None
    static_terminal: bool = False
    static_terminal_reason: str | None = None
    selected_by_recursive_policy: bool = False

    def as_observation(self) -> ResponseObservation:
        return ResponseObservation(
            move=self.move,
            uci=self.uci,
            san=self.san,
            probability=self.probability,
            response_value=self.value,
            selected_for_refinement=self.selected_for_refinement,
            depth_used=self.depth_used,
            static_value=self.static_value,
            refinement_rank=self.refinement_rank,
            used_shallow_landscape=self.used_shallow_landscape,
            conditional_probability=self.conditional_probability,
            branch_g_tilde=self.branch_g_tilde,
            branch_q_tilde=self.branch_q_tilde,
            branch_w_tilde=self.branch_w_tilde,
            branch_a_tilde=self.branch_a_tilde,
            branch_decomposition_error=self.branch_decomposition_error,
            static_components=self.static_components,
            board_fen=self.board_fen,
            static_terminal=self.static_terminal,
            static_terminal_reason=self.static_terminal_reason,
            selected_by_recursive_policy=self.selected_by_recursive_policy,
        )

    def as_dict(self) -> Dict[str, object]:
        data = self.as_observation().as_dict()
        data.update({
            "terminal_u": self.terminal_u,
            "branch_g_tilde": self.branch_g_tilde,
            "branch_q_tilde": self.branch_q_tilde,
            "branch_w_tilde": self.branch_w_tilde,
            "branch_a_tilde": self.branch_a_tilde,
            "branch_decomposition_error": self.branch_decomposition_error,
            "conditional_probability": self.conditional_probability,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ResponseEvaluation":
        obs = ResponseObservation.from_dict(data)
        return cls(
            move=obs.move,
            uci=obs.uci,
            san=obs.san,
            probability=obs.probability,
            value=obs.response_value,
            selected_for_refinement=obs.selected_for_refinement,
            static_value=obs.static_value,
            refinement_rank=obs.refinement_rank,
            depth_used=obs.depth_used,
            terminal_u=None if data.get("terminal_u") is None else float(data["terminal_u"]),
            branch_g_tilde=obs.branch_g_tilde,
            branch_q_tilde=obs.branch_q_tilde,
            branch_w_tilde=obs.branch_w_tilde,
            branch_a_tilde=obs.branch_a_tilde,
            branch_decomposition_error=obs.branch_decomposition_error,
            conditional_probability=obs.conditional_probability,
            used_shallow_landscape=obs.used_shallow_landscape,
            static_components=obs.static_components,
            board_fen=obs.board_fen,
            static_terminal=obs.static_terminal,
            static_terminal_reason=obs.static_terminal_reason,
            selected_by_recursive_policy=obs.selected_by_recursive_policy,
        )


@dataclass(frozen=True)
class MoveEvaluation:
    move: chess.Move
    san: str
    uci: str
    probability: float
    potential: float
    branch_value: float
    g_tilde: float | None
    selected_for_refinement: bool
    static_value_after_move: float | None
    base_potential: float
    phase_potential: float
    features: Dict[str, float]
    raw_probability: float | None = None
    considered_probability: float | None = None
    in_consideration_set: bool = True
    consideration_rank: int | None = None
    static_components_after_move: Dict[str, float] | None = None
    static_terminal_after_move: bool = False
    static_terminal_reason_after_move: str | None = None
    terminal_u: float | None = None
    q_tilde: float | None = None
    thermo_g_tilde: float | None = None
    w_tilde: float | None = None
    a_tilde: float | None = None
    retained_probability_mass: float | None = None
    retained_response_count: int | None = None
    total_probability_mass: float | None = None
    deepened_probability_mass: float | None = None
    nondeepened_probability_mass: float | None = None
    thermo_decomposition_error: float | None = None
    response_entropy: float | None = None
    response_N_eff: float | None = None
    response_K: int | None = None
    responses: Tuple[ResponseEvaluation, ...] = ()
    depth_used: int = 0
    cdepth: int = 0
    requested_cycles: int = 0
    requested_recursive_plies: int = 0
    recursive_plies_used: int = 0
    deepened_cycles: int = 0
    principal_variation: Tuple[str, ...] = ()
    endpoint_fen: str | None = None
    search_endpoint_fen: str | None = None
    thermodynamic_endpoint_fen: str | None = None
    qwa_unavailable_reason: str | None = None

    def as_branch_observation(self, *, remaining_plies: int, search_mode: SearchMode) -> LandscapeObservation:
        return LandscapeObservation(
            move=self.move,
            uci=self.uci,
            probability=self.probability,
            observable_value=self.branch_value,
            was_deepened=self.selected_for_refinement,
            depth_used=self.depth_used,
            response_neff=self.response_N_eff,
            response_k=self.response_K,
            total_probability_mass=self.total_probability_mass,
            deepened_probability_mass=self.deepened_probability_mass,
            nondeepened_probability_mass=self.nondeepened_probability_mass,
            response_branches=tuple(response.as_observation() for response in self.responses),
            search_mode=search_mode,
            remaining_plies=remaining_plies,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "san": self.san,
            "uci": self.uci,
            "probability": self.probability,
            "raw_probability": self.raw_probability if self.raw_probability is not None else self.probability,
            "considered_probability": self.considered_probability if self.considered_probability is not None else self.probability,
            "in_consideration_set": self.in_consideration_set,
            "consideration_rank": self.consideration_rank,
            "phi": self.potential,
            "potential": self.potential,
            "static_after": self.static_value_after_move,
            "static_value_after_move": self.static_value_after_move,
            "static_components_after_move": self.static_components_after_move,
            "static_terminal_after_move": self.static_terminal_after_move,
            "static_terminal_reason_after_move": self.static_terminal_reason_after_move,
            "terminal_u": self.terminal_u,
            "branch_value": self.branch_value,
            "g_tilde": self.g_tilde,
            "q_tilde": self.q_tilde,
            "thermo_g_tilde": self.thermo_g_tilde,
            "w_tilde": self.w_tilde,
            "a_tilde": self.a_tilde,
            "retained_probability_mass": self.retained_probability_mass,
            "retained_response_count": self.retained_response_count,
            "total_probability_mass": self.total_probability_mass,
            "deepened_probability_mass": self.deepened_probability_mass,
            "nondeepened_probability_mass": self.nondeepened_probability_mass,
            "thermo_decomposition_error": self.thermo_decomposition_error,
            "selected_for_refinement": self.selected_for_refinement,
            "actually_deepened": self.depth_used > 0,
            "base_potential": self.base_potential,
            "phase_potential": self.phase_potential,
            "total_potential": self.potential,
            "features": self.features,
            "response_entropy": self.response_entropy,
            "response_N_eff": self.response_N_eff,
            "response_K": self.response_K,
            "responses": [response.as_dict() for response in self.responses],
            "cdepth": self.cdepth,
            "requested_cycles": self.requested_cycles,
            "requested_recursive_plies": self.requested_recursive_plies,
            "recursive_plies_used": self.recursive_plies_used,
            "deepened_cycles": self.deepened_cycles,
            "principal_variation": list(self.principal_variation),
            "endpoint_fen": self.endpoint_fen,
            "search_endpoint_fen": self.search_endpoint_fen,
            "thermodynamic_endpoint_fen": self.thermodynamic_endpoint_fen,
            "qwa_unavailable_reason": self.qwa_unavailable_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "MoveEvaluation":
        uci = str(data["uci"])
        cdepth = int(data.get("cdepth", data.get("requested_cycles", 0)))
        requested_recursive_plies = int(data.get("requested_recursive_plies", data.get("requested_depth", data.get("depth", 0))))
        if cdepth <= 0 and requested_recursive_plies > 0:
            cdepth = (requested_recursive_plies + 1) // 2
        if requested_recursive_plies <= 0 and cdepth > 0:
            requested_recursive_plies = requested_recursive_plies_for_cdepth(cdepth)
        return cls(
            move=_move_from_uci(uci),
            san=str(data.get("san", "")),
            uci=uci,
            probability=float(data["probability"]),
            potential=float(data.get("potential", data.get("phi", 0.0))),
            branch_value=float(data.get("branch_value", 0.0)),
            g_tilde=None if data.get("g_tilde") is None else float(data["g_tilde"]),
            selected_for_refinement=bool(data.get("selected_for_refinement", False)),
            static_value_after_move=None if data.get("static_value_after_move", data.get("static_after")) is None else float(data.get("static_value_after_move", data.get("static_after"))),
            base_potential=float(data.get("base_potential", 0.0)),
            phase_potential=float(data.get("phase_potential", 0.0)),
            features=dict(data.get("features", {})),
            raw_probability=None if data.get("raw_probability") is None else float(data["raw_probability"]),
            considered_probability=None if data.get("considered_probability") is None else float(data["considered_probability"]),
            in_consideration_set=bool(data.get("in_consideration_set", True)),
            consideration_rank=None if data.get("consideration_rank") is None else int(data["consideration_rank"]),
            static_components_after_move=None if data.get("static_components_after_move") is None else {str(key): float(value) for key, value in dict(data["static_components_after_move"]).items()},
            static_terminal_after_move=bool(data.get("static_terminal_after_move", False)),
            static_terminal_reason_after_move=None if data.get("static_terminal_reason_after_move") is None else str(data["static_terminal_reason_after_move"]),
            terminal_u=None if data.get("terminal_u") is None else float(data["terminal_u"]),
            q_tilde=None if data.get("q_tilde") is None else float(data["q_tilde"]),
            thermo_g_tilde=None if data.get("thermo_g_tilde") is None else float(data["thermo_g_tilde"]),
            w_tilde=None if data.get("w_tilde") is None else float(data["w_tilde"]),
            a_tilde=None if data.get("a_tilde") is None else float(data["a_tilde"]),
            retained_probability_mass=None if data.get("retained_probability_mass") is None else float(data["retained_probability_mass"]),
            retained_response_count=None if data.get("retained_response_count") is None else int(data["retained_response_count"]),
            total_probability_mass=None if data.get("total_probability_mass") is None else float(data["total_probability_mass"]),
            deepened_probability_mass=None if data.get("deepened_probability_mass") is None else float(data["deepened_probability_mass"]),
            nondeepened_probability_mass=None if data.get("nondeepened_probability_mass") is None else float(data["nondeepened_probability_mass"]),
            thermo_decomposition_error=None if data.get("thermo_decomposition_error") is None else float(data["thermo_decomposition_error"]),
            response_entropy=None if data.get("response_entropy") is None else float(data["response_entropy"]),
            response_N_eff=None if data.get("response_N_eff") is None else float(data["response_N_eff"]),
            response_K=None if data.get("response_K") is None else int(data["response_K"]),
            responses=tuple(ResponseEvaluation.from_dict(item) for item in data.get("responses", [])),
            depth_used=int(data.get("recursive_plies_used", data.get("depth_used", 0))),
            cdepth=cdepth,
            requested_cycles=int(data.get("requested_cycles", cdepth)),
            requested_recursive_plies=requested_recursive_plies,
            recursive_plies_used=int(data.get("recursive_plies_used", data.get("depth_used_plies", data.get("depth_used", 0)))),
            deepened_cycles=int(data.get("deepened_cycles", 0)),
            principal_variation=tuple(str(uci) for uci in data.get("principal_variation", [])),
            endpoint_fen=None if data.get("endpoint_fen") is None else str(data["endpoint_fen"]),
            search_endpoint_fen=None if data.get("search_endpoint_fen") is None else str(data["search_endpoint_fen"]),
            thermodynamic_endpoint_fen=None if data.get("thermodynamic_endpoint_fen") is None else str(data["thermodynamic_endpoint_fen"]),
            qwa_unavailable_reason=None if data.get("qwa_unavailable_reason") is None else str(data["qwa_unavailable_reason"]),
        )


@dataclass(frozen=True)
class SearchResult:
    board_fen: str
    player: str
    side: str
    cdepth: int
    requested_recursive_plies: int
    search_mode: SearchMode
    beta: float
    deepening_quantile: float
    U: float
    entropy: float
    N_eff: float
    K: int
    selected_uci: Tuple[str, ...]
    moves: Tuple[MoveEvaluation, ...]
    diagnostics: Dict[str, float | int | str]
    kappa: float = KAPPA
    evaluation_weights: Dict[str, float] | None = None

    def branch_observations(self) -> Tuple[LandscapeObservation, ...]:
        return tuple(
            move.as_branch_observation(
                remaining_plies=self.requested_recursive_plies,
                search_mode=self.search_mode,
            )
            for move in self.moves
        )

    def as_node_selection(self) -> "NodeSelection":
        return NodeSelection(
            entropy=self.entropy,
            effective_moves=self.N_eff,
            expanded_count=self.K,
            selected_uci=self.selected_uci,
            branches=self.branch_observations(),
        )

    def as_viewer_panel(self, board_turn: chess.Color) -> Dict[str, object]:
        moves = [move.as_dict() for move in self.moves]
        moves.sort(key=lambda item: item["probability"], reverse=True)
        sort_direction = "probability_descending"
        for rank, move in enumerate(moves, start=1):
            move["advantage_rank"] = rank
            move["refinement_rank"] = rank
        return {
            "actual_decision": True,
            "player": self.player,
            "side": self.side,
            "expected_value": self.U,
            "entropy": self.entropy,
            "effective_moves": self.N_eff,
            "expanded_count": self.K,
            "legal_move_count": self.diagnostics.get("legal_move_count"),
            "consideration_count": self.K,
            "deepened_move_count": len(self.selected_uci),
            "cdepth": self.cdepth,
            "requested_cycles": self.cdepth,
            "requested_recursive_plies": self.requested_recursive_plies,
            "search_mode": self.search_mode,
            "deepening_quantile": self.deepening_quantile,
            "kappa": self.kappa,
            "temperature": 1.0 / (self.kappa * self.beta),
            "diagnostics": self.diagnostics,
            "sort_direction": sort_direction,
            "moves": moves,
        }

    def as_dict(self) -> Dict[str, object]:
        return {
            "board_fen": self.board_fen,
            "player": self.player,
            "side": self.side,
            "cdepth": self.cdepth,
            "requested_cycles": self.cdepth,
            "requested_recursive_plies": self.requested_recursive_plies,
            "search_mode": self.search_mode,
            "beta": self.beta,
            "kappa": self.kappa,
            "temperature": 1.0 / (self.kappa * self.beta),
            "deepening_quantile": self.deepening_quantile,
            "U": self.U,
            "entropy": self.entropy,
            "N_eff": self.N_eff,
            "K": self.K,
            "legal_move_count": self.diagnostics.get("legal_move_count"),
            "consideration_count": self.K,
            "deepened_move_count": len(self.selected_uci),
            "selected_uci": list(self.selected_uci),
            "moves": [move.as_dict() for move in self.moves],
            "diagnostics": self.diagnostics,
            "evaluation_weights": self.evaluation_weights,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SearchResult":
        cdepth = int(data.get("cdepth", data.get("requested_cycles", 0)))
        requested_recursive_plies = int(data.get("requested_recursive_plies", data.get("depth", 0)))
        if cdepth <= 0 and requested_recursive_plies > 0:
            cdepth = (requested_recursive_plies + 1) // 2
        if requested_recursive_plies <= 0 and cdepth > 0:
            requested_recursive_plies = requested_recursive_plies_for_cdepth(cdepth)
        return cls(
            board_fen=str(data["board_fen"]),
            player=str(data.get("player", "")),
            side=str(data.get("side", "")),
            cdepth=cdepth,
            requested_recursive_plies=requested_recursive_plies,
            search_mode=_validate_search_mode(str(data.get("search_mode", "accurate"))),
            beta=float(data.get("beta", 1.0)),
            kappa=float(data.get("kappa", KAPPA)),
            deepening_quantile=float(data.get("deepening_quantile", data.get("adaptive_c", 0.8))),
            U=float(data.get("U", data.get("expected_value", 0.0))),
            entropy=float(data.get("entropy", 0.0)),
            N_eff=float(data.get("N_eff", data.get("effective_moves", 0.0))),
            K=int(data.get("K", data.get("expanded_count", 0))),
            selected_uci=tuple(str(uci) for uci in data.get("selected_uci", [])),
            moves=tuple(MoveEvaluation.from_dict(item) for item in data.get("moves", [])),
            diagnostics=dict(data.get("diagnostics", {})),
            evaluation_weights=None if data.get("evaluation_weights") is None else {str(key): float(value) for key, value in dict(data["evaluation_weights"]).items()},
        )


@dataclass(frozen=True)
class NodeSelection:
    entropy: float
    effective_moves: float
    expanded_count: int
    selected_uci: Tuple[str, ...]
    branches: Tuple[LandscapeObservation, ...]


@dataclass(frozen=True)
class RecursiveResult:
    value: float
    endpoint_fen: str
    principal_variation: Tuple[str, ...]
    depth_reached: int
    terminal_reason: str | None = None


@dataclass
class SearchDiagnostics:
    cdepth: int
    requested_recursive_plies: int
    deepening_quantile: float
    search_mode: SearchMode = "accurate"
    search_workers: int = 1
    parallel_min_branches: int = 0
    nodes_evaluated: int = 0
    static_evaluations: int = 0
    static_value_requests: int = 0
    scalar_evaluations: int = 0
    combined_evaluations: int = 0
    static_component_computations: int = 0
    board_context_calls: int = 0
    board_context_requests: int = 0
    board_context_computations: int = 0
    board_context_cache_hits: int = 0
    board_context_cache_entries: int = 0
    board_context_cache_peak_entries: int = 0
    center_control_raw_computations: int = 0
    king_safety_raw_computations: int = 0
    development_slots_computations: int = 0
    king_safe_squares_computations: int = 0
    legal_mobility_computations: int = 0
    legal_mobility_requests: int = 0
    legal_mobility_direct_counts: int = 0
    legal_mobility_probe_counts: int = 0
    legal_mobility_board_copies: int = 0
    pawn_structure_counts_computations: int = 0
    move_distribution_calls: int = 0
    recursive_nodes: int = 0
    maximum_depth_reached: int = 0
    cache_hits: int = 0
    expanded_total: int = 0
    expanded_nodes: int = 0
    response_expanded_total: int = 0
    response_expanded_nodes: int = 0
    parallel_branches: int = 0
    search_result_reuse_count: int = 0
    root_candidate_branch_evaluations: int = 0
    root_candidate_recursive_evaluations: int = 0
    adaptive_depth_zero_nodes: int = 0
    adaptive_depth_one_cycle_nodes: int = 0
    adaptive_depth_two_cycle_nodes: int = 0
    adaptive_depth_full_nodes: int = 0
    adaptive_depth_reductions: int = 0
    forced_continuation_nodes: int = 0
    objective_forced_nodes: int = 0
    subjective_forced_nodes: int = 0
    quantile_forced_nodes: int = 0
    elapsed_time: float = 0.0

    @property
    def average_k(self) -> float:
        if self.expanded_nodes == 0:
            return 0.0
        return self.expanded_total / self.expanded_nodes

    @property
    def average_response_k(self) -> float:
        if self.response_expanded_nodes == 0:
            return 0.0
        return self.response_expanded_total / self.response_expanded_nodes

    def add_child(self, child: "SearchDiagnostics") -> None:
        self.nodes_evaluated += child.nodes_evaluated
        self.static_evaluations += child.static_evaluations
        self.static_value_requests += child.static_value_requests
        self.scalar_evaluations += child.scalar_evaluations
        self.combined_evaluations += child.combined_evaluations
        self.static_component_computations += child.static_component_computations
        self.board_context_calls += child.board_context_calls
        self.board_context_requests += child.board_context_requests
        self.board_context_computations += child.board_context_computations
        self.board_context_cache_hits += child.board_context_cache_hits
        self.board_context_cache_entries += child.board_context_cache_entries
        self.board_context_cache_peak_entries = max(
            self.board_context_cache_peak_entries,
            child.board_context_cache_peak_entries,
        )
        self.center_control_raw_computations += child.center_control_raw_computations
        self.king_safety_raw_computations += child.king_safety_raw_computations
        self.development_slots_computations += child.development_slots_computations
        self.king_safe_squares_computations += child.king_safe_squares_computations
        self.legal_mobility_computations += child.legal_mobility_computations
        self.legal_mobility_requests += child.legal_mobility_requests
        self.legal_mobility_direct_counts += child.legal_mobility_direct_counts
        self.legal_mobility_probe_counts += child.legal_mobility_probe_counts
        self.legal_mobility_board_copies += child.legal_mobility_board_copies
        self.pawn_structure_counts_computations += child.pawn_structure_counts_computations
        self.move_distribution_calls += child.move_distribution_calls
        self.recursive_nodes += child.recursive_nodes
        self.maximum_depth_reached = max(self.maximum_depth_reached, child.maximum_depth_reached)
        self.cache_hits += child.cache_hits
        self.expanded_total += child.expanded_total
        self.expanded_nodes += child.expanded_nodes
        self.response_expanded_total += child.response_expanded_total
        self.response_expanded_nodes += child.response_expanded_nodes
        self.parallel_branches += child.parallel_branches
        self.search_result_reuse_count += child.search_result_reuse_count
        self.root_candidate_branch_evaluations += child.root_candidate_branch_evaluations
        self.root_candidate_recursive_evaluations += child.root_candidate_recursive_evaluations
        self.adaptive_depth_zero_nodes += child.adaptive_depth_zero_nodes
        self.adaptive_depth_one_cycle_nodes += child.adaptive_depth_one_cycle_nodes
        self.adaptive_depth_two_cycle_nodes += child.adaptive_depth_two_cycle_nodes
        self.adaptive_depth_full_nodes += child.adaptive_depth_full_nodes
        self.adaptive_depth_reductions += child.adaptive_depth_reductions
        self.forced_continuation_nodes += child.forced_continuation_nodes
        self.objective_forced_nodes += child.objective_forced_nodes
        self.subjective_forced_nodes += child.subjective_forced_nodes
        self.quantile_forced_nodes += child.quantile_forced_nodes

    def as_dict(self) -> Dict[str, float | int | str]:
        result = asdict(self)
        result["average_k"] = self.average_k
        result["average_response_k"] = self.average_response_k
        return result


class _CachedEvaluator:
    """StaticEvaluator-compatible proxy owned by one adaptive search."""

    def __init__(self, search: "AdaptiveExpectedValue") -> None:
        self.search = search
        self.weights = search.evaluator.weights

    def evaluate(self, board: chess.Board, context: BoardFeatureContext | None = None) -> float:
        return self.search.static_value(board, context=context)

    def evaluate_with_components(
        self, board: chess.Board, context: BoardFeatureContext | None = None
    ) -> tuple[float, Dict[str, float] | None]:
        return self.search.static_value_with_components(board, context=context)

    def components(
        self, board: chess.Board, context: BoardFeatureContext | None = None
    ) -> Dict[str, float] | None:
        components = self.search.evaluator.components(board, context=context)
        if components is not None:
            self.search.diagnostics.static_component_computations += 1
        return components

    def terminal_reason(self, board: chess.Board) -> str | None:
        return self.search.evaluator.terminal_reason(board)


def _terminal_observation(value: float, remaining_plies: int, search_mode: SearchMode) -> NodeSelection:
    branch = LandscapeObservation(
        move=chess.Move.null(),
        uci="__terminal__",
        probability=1.0,
        observable_value=value,
        was_deepened=False,
        depth_used=0,
        search_mode=search_mode,
        remaining_plies=remaining_plies,
    )
    return NodeSelection(0.0, 1.0, 0, (), (branch,))


def mass_preserving_expectation(branches: Iterable[LandscapeObservation]) -> float:
    """Expectation over original branch probabilities, used only for diagnostics."""
    return sum(branch.probability * branch.observable_value for branch in branches)


def side_to_move_backup(
    side_to_move: chess.Color,
    branches: Iterable[LandscapeObservation],
) -> float:
    """Back up available branch endpoint values in the White-positive frame."""
    values = [branch.observable_value for branch in branches]
    if not values:
        raise ValueError("cannot back up an empty branch set")
    return max(values) if side_to_move == chess.WHITE else min(values)


def _branch_observation_worker(args: Dict[str, object]) -> tuple[LandscapeObservation, Dict[str, float | int | str]]:
    evaluator = StaticEvaluator(args["eval_weights"])  # type: ignore[arg-type]
    search = AdaptiveExpectedValue(
        args["style"],  # type: ignore[arg-type]
        float(args["beta"]),
        evaluator,
        cdepth=int(args["cdepth"]),
        deepening_quantile=float(args["deepening_quantile"]),
        search_mode=_validate_search_mode(str(args["search_mode"])),
        search_workers=1,
        parallel_min_branches=int(args["parallel_min_branches"]),
        kappa=float(args.get("kappa", KAPPA)),
    )
    search.begin_diagnostics()
    board = chess.Board(str(args["fen"]))
    record = args["record"]  # type: ignore[assignment]
    branch = search._evaluate_branch(  # noqa: SLF001 - process worker for this class.
        board,
        record,  # type: ignore[arg-type]
        set(args["selected"]),  # type: ignore[arg-type]
        int(args["remaining_plies"]),
        int(args["level"]),
        None if args.get("node_shallow_value") is None else float(args["node_shallow_value"]),
    )
    return branch, search.finish_diagnostics()


def _branch_worker(args: Dict[str, object]) -> tuple[MoveEvaluation, LandscapeObservation, Dict[str, float | int | str]]:
    evaluator = StaticEvaluator(args["eval_weights"])  # type: ignore[arg-type]
    search = AdaptiveExpectedValue(
        args["style"],  # type: ignore[arg-type]
        float(args["beta"]),
        evaluator,
        cdepth=int(args["cdepth"]),
        deepening_quantile=float(args["deepening_quantile"]),
        search_mode=_validate_search_mode(str(args["search_mode"])),
        search_workers=1,
        parallel_min_branches=int(args["parallel_min_branches"]),
        kappa=float(args.get("kappa", KAPPA)),
    )
    search.begin_diagnostics()
    board = chess.Board(str(args["fen"]))
    record = args["record"]  # type: ignore[assignment]
    move, branch = search._evaluate_root_candidate_once(  # noqa: SLF001 - process worker for this class.
        board,
        record,  # type: ignore[arg-type]
        set(args["selected"]),  # type: ignore[arg-type]
        int(args["remaining_plies"]),
        int(args["level"]),
        None if args.get("node_shallow_value") is None else float(args["node_shallow_value"]),
        float(args["current_u"]),
        tuple(args["current_branches"]),  # type: ignore[arg-type]
        bool(args["detailed"]),
    )
    return move, branch, search.finish_diagnostics()


class AdaptiveExpectedValue:
    """Compute subjective minimax values over a cycle-depth horizon."""

    def __init__(
        self,
        style: Style,
        beta: float,
        evaluator: StaticEvaluator,
        cdepth: int = 2,
        deepening_quantile: float = 0.8,
        adaptive_c: float | None = None,
        search_mode: SearchMode = "accurate",
        search_workers: int | None = 1,
        parallel_min_branches: int = 8,
        kappa: float = KAPPA,
        use_context_cache: bool = True,
    ) -> None:
        cdepth = validate_cdepth(cdepth)
        requested_recursive_plies = requested_recursive_plies_for_cdepth(cdepth)
        if adaptive_c is not None:
            raise ValueError("adaptive_c is deprecated; use deepening_quantile instead")
        deepening_quantile = validate_deepening_quantile(deepening_quantile)
        if kappa <= 0.0:
            raise ValueError("kappa must be greater than 0")
        if search_workers is None:
            search_workers = max(1, (os.cpu_count() or 1) - 1)
        if search_workers < 1:
            raise ValueError("search_workers must be at least 1")
        if parallel_min_branches < 1:
            raise ValueError("parallel_min_branches must be at least 1")
        self.style = style
        self.beta = beta
        self.kappa = kappa
        self.evaluator = evaluator
        self.cdepth = cdepth
        self.requested_recursive_plies = requested_recursive_plies
        self.deepening_quantile = deepening_quantile
        self.search_mode = _validate_search_mode(search_mode)
        self.search_workers = search_workers
        self.parallel_min_branches = parallel_min_branches
        self.use_context_cache = use_context_cache
        self._static_cache: Dict[PositionKey, float] = {}
        self._context_cache: Dict[PositionKey, BoardFeatureContext] = {}
        self._landscape_cache: Dict[PositionKey, MoveLandscape] = {}
        self._response_landscape_cache: Dict[PositionKey, MoveLandscape] = {}
        self._value_cache: Dict[CacheKey, float] = {}
        self._selection_cache: Dict[CacheKey, NodeSelection] = {}
        self._future_value_cache: Dict[CacheKey, float] = {}
        self._future_selection_cache: Dict[CacheKey, NodeSelection] = {}
        self._recursive_result_cache: Dict[CacheKey, RecursiveResult] = {}
        self._result_cache: Dict[Tuple[PositionKey, int, str], SearchResult] = {}
        self._cached_evaluator = _CachedEvaluator(self)
        self._branch_cache_lifecycle: list[Dict[str, Dict[str, int]]] = []
        self.diagnostics = SearchDiagnostics(
            cdepth,
            requested_recursive_plies,
            deepening_quantile,
            self.search_mode,
            self.search_workers,
            self.parallel_min_branches,
        )
        self._started_at = 0.0

    def clear_transient_caches(self) -> None:
        """Release root-search data that should not survive an actual turn."""
        self._static_cache.clear()
        self._context_cache.clear()
        self.diagnostics.board_context_cache_entries = 0
        self._landscape_cache.clear()
        self._response_landscape_cache.clear()
        self._value_cache.clear()
        self._selection_cache.clear()
        self._future_value_cache.clear()
        self._future_selection_cache.clear()
        self._recursive_result_cache.clear()
        self._result_cache.clear()
        self._branch_cache_lifecycle.clear()
        self._qwa_root_board = None

    def clear_branch_caches(self) -> None:
        """Release recursive state for a completed root candidate branch."""
        self.clear_transient_caches()

    def _cache_sizes(self) -> Dict[str, int]:
        return {
            "static": len(self._static_cache),
            "context": len(self._context_cache),
            "landscape": len(self._landscape_cache),
            "response_landscape": len(self._response_landscape_cache),
            "value": len(self._value_cache),
            "selection": len(self._selection_cache),
            "future_value": len(self._future_value_cache),
            "future_selection": len(self._future_selection_cache),
            "recursive_result": len(self._recursive_result_cache),
            "result": len(self._result_cache),
        }

    def _new_branch_context(self) -> "AdaptiveExpectedValue":
        return AdaptiveExpectedValue(
            self.style,
            self.beta,
            self.evaluator,
            cdepth=self.cdepth,
            deepening_quantile=self.deepening_quantile,
            search_mode=self.search_mode,
            search_workers=1,
            parallel_min_branches=self.parallel_min_branches,
            kappa=self.kappa,
            use_context_cache=self.use_context_cache,
        )

    @staticmethod
    def position_key(board: chess.Board) -> PositionKey:
        return (
            board.fen(en_passant="fen"),
            board.is_repetition(2),
            board.is_repetition(3),
            board.is_fivefold_repetition(),
        )

    def begin_diagnostics(self) -> None:
        self.diagnostics = SearchDiagnostics(
            self.cdepth,
            self.requested_recursive_plies,
            self.deepening_quantile,
            self.search_mode,
            self.search_workers,
            self.parallel_min_branches,
        )
        self._started_at = time.perf_counter()

    def finish_diagnostics(self) -> Dict[str, float | int | str]:
        self.diagnostics.elapsed_time = time.perf_counter() - self._started_at
        return self.diagnostics.as_dict()

    def _get_board_context(
        self, board: chess.Board, key: PositionKey | None = None
    ) -> BoardFeatureContext:
        position_key = key or self.position_key(board)
        self.diagnostics.board_context_requests += 1
        if self.use_context_cache:
            cached = self._context_cache.get(position_key)
            if cached is not None:
                self.diagnostics.cache_hits += 1
                self.diagnostics.board_context_cache_hits += 1
                return cached
        context = board_context(board, stats=self.diagnostics.__dict__)
        if self.use_context_cache:
            self._context_cache[position_key] = context
            entries = len(self._context_cache)
            self.diagnostics.board_context_cache_entries = entries
            self.diagnostics.board_context_cache_peak_entries = max(
                self.diagnostics.board_context_cache_peak_entries,
                entries,
            )
        self.diagnostics.board_context_calls += 1
        self.diagnostics.board_context_computations += 1
        return context

    def board_context(
        self, board: chess.Board, key: PositionKey | None = None
    ) -> BoardFeatureContext:
        return self._get_board_context(board, key)

    def static_value(
        self,
        board: chess.Board,
        key: PositionKey | None = None,
        context: BoardFeatureContext | None = None,
    ) -> float:
        self.diagnostics.static_value_requests += 1
        self.diagnostics.scalar_evaluations += 1
        position_key = key or self.position_key(board)
        if position_key in self._static_cache:
            self.diagnostics.cache_hits += 1
            return self._static_cache[position_key]
        values = context or self.board_context(board, position_key)
        value, components = self.evaluator.evaluate_with_components(board, context=values)
        self._static_cache[position_key] = value
        self.diagnostics.static_evaluations += 1
        if components is not None:
            self.diagnostics.static_component_computations += 1
        return value

    def static_value_with_components(
        self,
        board: chess.Board,
        key: PositionKey | None = None,
        context: BoardFeatureContext | None = None,
    ) -> tuple[float, Dict[str, float] | None]:
        self.diagnostics.static_value_requests += 1
        self.diagnostics.combined_evaluations += 1
        position_key = key or self.position_key(board)
        values = context or self.board_context(board, position_key)
        if position_key in self._static_cache:
            self.diagnostics.cache_hits += 1
            components = self.evaluator.components(board, context=values)
            if components is not None:
                self.diagnostics.static_component_computations += 1
            return self._static_cache[position_key], components
        value, components = self.evaluator.evaluate_with_components(board, context=values)
        self._static_cache[position_key] = value
        self.diagnostics.static_evaluations += 1
        if components is not None:
            self.diagnostics.static_component_computations += 1
        return value, components

    def landscape(
        self, board: chess.Board, key: PositionKey | None = None
    ) -> MoveLandscape:
        position_key = key or self.position_key(board)
        if position_key in self._landscape_cache:
            self.diagnostics.cache_hits += 1
            return self._landscape_cache[position_key]
        context = self.board_context(board, position_key)
        self.diagnostics.move_distribution_calls += 1
        landscape = move_distribution(
            board,
            self.style,
            self.beta,
            self._cached_evaluator,
            before_context=context,
            context_provider=self.board_context,
            kappa=self.kappa,
        )
        self._landscape_cache[position_key] = landscape
        return landscape

    def response_landscape(
        self, board: chess.Board, key: PositionKey | None = None
    ) -> MoveLandscape:
        position_key = key or self.position_key(board)
        if position_key in self._response_landscape_cache:
            self.diagnostics.cache_hits += 1
            return self._response_landscape_cache[position_key]
        context = self.board_context(board, position_key)
        self.diagnostics.move_distribution_calls += 1
        landscape = move_distribution(
            board,
            self.style,
            self.beta,
            self._cached_evaluator,
            before_context=context,
            context_provider=self.board_context,
            kappa=self.kappa,
        )
        self._response_landscape_cache[position_key] = landscape
        return landscape

    def _record_adaptive_depth_decision(
        self,
        *,
        cycle_limit: int | None,
        inherited_remaining: int,
        effective_remaining: int,
        legal_move_count: int,
        consideration_count: int,
        deepening_branch_count: int,
    ) -> None:
        if cycle_limit is None:
            self.diagnostics.forced_continuation_nodes += 1
            if legal_move_count == 1:
                self.diagnostics.objective_forced_nodes += 1
            elif consideration_count == 1:
                self.diagnostics.subjective_forced_nodes += 1
            elif deepening_branch_count == 1:
                self.diagnostics.quantile_forced_nodes += 1
        elif cycle_limit == 0:
            self.diagnostics.adaptive_depth_zero_nodes += 1
        elif cycle_limit == 1:
            self.diagnostics.adaptive_depth_one_cycle_nodes += 1
        elif cycle_limit == 2:
            self.diagnostics.adaptive_depth_two_cycle_nodes += 1
        else:
            self.diagnostics.adaptive_depth_full_nodes += 1
        if effective_remaining < inherited_remaining:
            self.diagnostics.adaptive_depth_reductions += 1

    def _effective_remaining_plies(
        self,
        deepening_branch_count: int,
        remaining_plies: int,
        level: int,
    ) -> tuple[int, int | None, int]:
        cycle_limit = adaptive_cycle_limit(deepening_branch_count, self.cdepth)
        if cycle_limit is None:
            locally_allowed = remaining_plies
        else:
            locally_allowed = adaptive_recursive_plies_from_node(cycle_limit, level)
        effective_remaining = min(remaining_plies, locally_allowed)
        return effective_remaining, cycle_limit, locally_allowed

    def node_selection(
        self,
        board: chess.Board,
        remaining_plies: int | None = None,
        level: int = 0,
    ) -> NodeSelection | None:
        remaining = self.requested_recursive_plies if remaining_plies is None else remaining_plies
        position_key = self.position_key(board)
        phase = level % 2
        cache_key = (position_key, remaining, phase)
        selection = self._selection_cache.get(cache_key)
        if selection is None and remaining >= 0:
            self._expected_value(board, remaining, level)
            selection = self._selection_cache.get(cache_key)
            if selection is None and not board.is_game_over(claim_draw=False):
                landscape = self.landscape(board, position_key)
                considered = tuple(record for record in landscape.records if record.in_consideration_set)
                ranked = ranked_for_refinement(considered, board.turn)
                expanded_count = (
                    0
                    if remaining == 0
                    else deepening_count_for_quantile(
                        (record.probability for record in ranked),
                        self.deepening_quantile,
                    )
                )
                effective_remaining, _, _ = self._effective_remaining_plies(expanded_count, remaining, level)
                selection = self._selection_cache.get((position_key, effective_remaining, phase))
        return selection

    def branch_observations(
        self, board: chess.Board, remaining_plies: int | None = None, level: int = 0
    ) -> Tuple[LandscapeObservation, ...]:
        remaining = self.requested_recursive_plies if remaining_plies is None else remaining_plies
        self._expected_value(board, remaining, level)
        selection = self.node_selection(board, remaining, level)
        return selection.branches if selection is not None else ()

    def expected_value(self, board: chess.Board, remaining_plies: int | None = None) -> float:
        remaining = self.requested_recursive_plies if remaining_plies is None else remaining_plies
        if remaining < 0:
            raise ValueError("remaining_plies must be at least 0")
        return self._expected_value(board, remaining, 0)

    def _recursive_result(
        self, board: chess.Board, remaining_depth: int, level: int
    ) -> RecursiveResult:
        if remaining_depth < 0:
            raise ValueError("remaining_depth must be non-negative")
        cache_key = (self.position_key(board), remaining_depth, level % 2)
        cached = self._recursive_result_cache.get(cache_key)
        if cached is not None:
            self.diagnostics.cache_hits += 1
            return cached

        self.diagnostics.recursive_nodes += 1
        value = self._expected_value(board, remaining_depth, level)
        if board.is_game_over(claim_draw=False):
            result = RecursiveResult(
                value=value,
                endpoint_fen=board.fen(),
                principal_variation=(),
                depth_reached=0,
                terminal_reason=self.evaluator.terminal_reason(board) or "terminal",
            )
            self._recursive_result_cache[cache_key] = result
            return result

        selection = self.node_selection(board, remaining_depth, level)
        if remaining_depth == 0 or selection is None or not selection.selected_uci:
            result = RecursiveResult(
                value=value,
                endpoint_fen=board.fen(),
                principal_variation=(),
                depth_reached=0,
            )
            self._recursive_result_cache[cache_key] = result
            return result

        if not selection.branches:
            result = RecursiveResult(value=value, endpoint_fen=board.fen(), principal_variation=(), depth_reached=0)
            self._recursive_result_cache[cache_key] = result
            return result

        chooser = max if board.turn == chess.WHITE else min
        chosen = chooser(selection.branches, key=lambda branch: branch.observable_value)
        child = board.copy(stack=False)
        child.push(chosen.move)
        if chosen.was_deepened:
            child_result = self._recursive_result(child, max(0, chosen.depth_used - 1), level + 1)
            result = RecursiveResult(
                value=chosen.observable_value,
                endpoint_fen=child_result.endpoint_fen,
                principal_variation=(chosen.uci,) + child_result.principal_variation,
                depth_reached=1 + child_result.depth_reached,
                terminal_reason=child_result.terminal_reason,
            )
        else:
            terminal_reason = self.evaluator.terminal_reason(child) if child.is_game_over(claim_draw=False) else None
            result = RecursiveResult(
                value=chosen.observable_value,
                endpoint_fen=child.fen(),
                principal_variation=(chosen.uci,),
                depth_reached=1,
                terminal_reason=terminal_reason,
            )
        self._recursive_result_cache[cache_key] = result
        return result

    def _future_subjective_value(self, board: chess.Board, remaining_depth: int, level: int) -> float:
        """Compatibility wrapper for the recursive value F_p(B, n)."""
        if remaining_depth < 0:
            raise ValueError("remaining_depth must be at least 0")
        return self._expected_value(board, remaining_depth, level)

    def _future_branch_observations(
        self, board: chess.Board, remaining_depth: int, level: int
    ) -> Tuple[LandscapeObservation, ...]:
        self._expected_value(board, remaining_depth, level)
        selection = self.node_selection(board, remaining_depth, level)
        return selection.branches if selection is not None else ()

    def _principal_variation_cycle_metadata(
        self,
        root_board: chess.Board,
        principal_variation: Tuple[str, ...],
        search_endpoint_fen: str,
    ) -> Dict[str, object]:
        board = root_board.copy(stack=False)
        root_turn = root_board.turn
        recursive_plies_used = 0
        deepened_cycles = 0
        thermodynamic_endpoint_fen: str | None = None
        for uci in principal_variation:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
            board.push(move)
            recursive_plies_used += 1
            if board.turn == root_turn and recursive_plies_used % 2 == 0:
                if board.is_game_over(claim_draw=False):
                    continue
                deepened_cycles = recursive_plies_used // 2
                thermodynamic_endpoint_fen = board.fen()
        return {
            "recursive_plies_used": recursive_plies_used,
            "deepened_cycles": deepened_cycles,
            "search_endpoint_fen": search_endpoint_fen,
            "thermodynamic_endpoint_fen": thermodynamic_endpoint_fen,
        }

    def _root_shallow_observations(self, landscape: MoveLandscape) -> Tuple[LandscapeObservation, ...]:
        return tuple(
            LandscapeObservation(
                move=record.move,
                uci=record.uci,
                probability=record.probability,
                observable_value=float(record.static_after),
                was_deepened=False,
                depth_used=0,
                search_mode=self.search_mode,
                remaining_plies=0,
            )
            for record in landscape.considered_records
        )

    def _outside_consideration_move(self, record: MoveRecord, current_u: float) -> MoveEvaluation:
        return MoveEvaluation(
            move=record.move,
            san=record.san,
            uci=record.uci,
            probability=record.probability,
            potential=record.phi,
            branch_value=current_u,
            selected_for_refinement=False,
            static_value_after_move=None,
            base_potential=record.base_potential,
            phase_potential=record.phase_potential,
            features=record.features,
            raw_probability=record.raw_probability,
            considered_probability=record.considered_probability,
            in_consideration_set=False,
            consideration_rank=record.consideration_rank,
            static_components_after_move=None,
            static_terminal_after_move=False,
            static_terminal_reason_after_move=None,
            terminal_u=None,
            g_tilde=None,
            cdepth=self.cdepth,
            requested_cycles=self.cdepth,
            requested_recursive_plies=self.requested_recursive_plies,
            qwa_unavailable_reason="outside_consideration_set",
        )

    def _materialize_root_candidate(
        self,
        record: MoveRecord,
        branch: LandscapeObservation | None,
        thermo: Dict[str, object],
    ) -> MoveEvaluation:
        branch_value = (
            float(thermo["terminal_u"])
            if thermo["terminal_u"] is not None
            else (branch.observable_value if branch else float(record.static_after or 0.0))
        )
        return MoveEvaluation(
            move=record.move,
            san=record.san,
            uci=record.uci,
            probability=record.probability,
            potential=record.phi,
            branch_value=branch_value,
            selected_for_refinement=branch.was_deepened if branch else False,
            static_value_after_move=record.static_after,
            base_potential=record.base_potential,
            phase_potential=record.phase_potential,
            features=record.features,
            raw_probability=record.raw_probability,
            considered_probability=record.considered_probability,
            in_consideration_set=record.in_consideration_set,
            consideration_rank=record.consideration_rank,
            static_components_after_move=record.static_components_after,
            static_terminal_after_move=record.static_terminal_after,
            static_terminal_reason_after_move=record.static_terminal_reason_after,
            terminal_u=thermo["terminal_u"],
            g_tilde=thermo["g_tilde"],
            q_tilde=thermo["q_tilde"],
            thermo_g_tilde=thermo["thermo_g_tilde"],
            w_tilde=thermo["w_tilde"],
            a_tilde=thermo["a_tilde"],
            retained_probability_mass=thermo["retained_probability_mass"],
            retained_response_count=thermo["retained_response_count"],
            total_probability_mass=thermo["total_probability_mass"],
            deepened_probability_mass=thermo["deepened_probability_mass"],
            nondeepened_probability_mass=thermo["nondeepened_probability_mass"],
            thermo_decomposition_error=thermo["thermo_decomposition_error"],
            response_entropy=thermo["response_entropy"],
            response_N_eff=thermo["response_N_eff"],
            response_K=thermo["response_K"],
            responses=thermo["responses"],
            depth_used=branch.depth_used if branch else 0,
            cdepth=self.cdepth,
            requested_cycles=self.cdepth,
            requested_recursive_plies=self.requested_recursive_plies,
            recursive_plies_used=thermo["recursive_plies_used"],
            deepened_cycles=thermo["deepened_cycles"],
            principal_variation=thermo["principal_variation"],
            endpoint_fen=thermo["endpoint_fen"],
            search_endpoint_fen=thermo["search_endpoint_fen"],
            thermodynamic_endpoint_fen=thermo["thermodynamic_endpoint_fen"],
            qwa_unavailable_reason=thermo["qwa_unavailable_reason"],
        )

    def _evaluate_root_candidate_once(
        self,
        board: chess.Board,
        record: MoveRecord,
        selected: set[str],
        remaining_plies: int,
        level: int,
        node_shallow_value: float | None,
        current_u: float,
        current_branches: Tuple[LandscapeObservation, ...],
        detailed: bool,
    ) -> tuple[MoveEvaluation, LandscapeObservation]:
        self.diagnostics.root_candidate_branch_evaluations += 1
        own_selected = record.uci in selected
        if own_selected:
            self.diagnostics.root_candidate_recursive_evaluations += 1
        branch_seed = LandscapeObservation(
            move=record.move,
            uci=record.uci,
            probability=record.probability,
            observable_value=float(record.static_after),
            was_deepened=own_selected,
            depth_used=remaining_plies if own_selected else 0,
            search_mode=self.search_mode,
            remaining_plies=remaining_plies,
        )
        thermo = self._candidate_thermodynamics(
            board,
            record,
            branch_seed,
            current_u,
            current_branches,
            detailed=detailed,
        )
        move = self._materialize_root_candidate(record, branch_seed, thermo)
        response_observations = tuple(response.as_observation() for response in move.responses)
        branch = LandscapeObservation(
            move=record.move,
            uci=record.uci,
            probability=record.probability,
            observable_value=move.branch_value,
            was_deepened=move.selected_for_refinement,
            depth_used=move.depth_used,
            response_neff=move.response_N_eff,
            response_k=move.response_K,
            total_probability_mass=move.total_probability_mass,
            deepened_probability_mass=move.deepened_probability_mass,
            nondeepened_probability_mass=move.nondeepened_probability_mass,
            response_branches=response_observations,
            search_mode=self.search_mode,
            remaining_plies=remaining_plies,
        )
        return move, branch

    def search_result(self, board: chess.Board, player: str = "", side: str | None = None) -> SearchResult:
        result_side = side or ("white" if board.turn == chess.WHITE else "black")
        key = (self.position_key(board), self.requested_recursive_plies, result_side)
        cached = self._result_cache.get(key)
        if cached is not None:
            self.diagnostics.search_result_reuse_count += 1
            return cached
        self._branch_cache_lifecycle.clear()
        landscape = self.landscape(board)
        current_u = landscape.expected_value
        current_branches = self._root_shallow_observations(landscape)
        considered = tuple(record for record in landscape.records if record.in_consideration_set)
        ranked = ranked_for_refinement(considered, board.turn)
        expanded_count = deepening_count_for_quantile(
            (record.probability for record in ranked),
            self.deepening_quantile,
        )
        selected_uci = tuple(record.uci for record in ranked[:expanded_count])
        root_cycle_limit = adaptive_cycle_limit(expanded_count, self.cdepth)
        root_forced_continuation = root_cycle_limit is None
        if root_cycle_limit is None:
            root_recursive_plies = self.requested_recursive_plies
        else:
            root_recursive_plies = adaptive_recursive_plies_for_root(root_cycle_limit)
        self._record_adaptive_depth_decision(
            cycle_limit=root_cycle_limit,
            inherited_remaining=self.requested_recursive_plies,
            effective_remaining=root_recursive_plies,
            legal_move_count=landscape.legal_move_count,
            consideration_count=landscape.consideration_count,
            deepening_branch_count=expanded_count,
        )
        selected = set(selected_uci)
        self.diagnostics.nodes_evaluated += 1
        self.diagnostics.expanded_total += expanded_count
        self.diagnostics.expanded_nodes += 1
        root_diagnostics = {
            **self.diagnostics.as_dict(),
            "legal_move_count": landscape.legal_move_count,
            "entropy": landscape.entropy,
            "n_eff": landscape.effective_moves,
            "consideration_count": landscape.consideration_count,
            "deepening_branch_count": expanded_count,
            "consideration_probability_mass_raw": landscape.consideration_probability_mass_raw,
            "deepening_quantile": self.deepening_quantile,
            "deepened_move_count": len(selected_uci),
            "deepening_probability_mass_considered": sum(
                record.probability for record in landscape.considered_records if record.uci in selected
            ),
        }
        moves_by_uci: dict[str, MoveEvaluation] = {}
        branch_by_uci: dict[str, LandscapeObservation] = {}
        detailed_uci = set(selected_uci)
        if self.search_workers > 1 and len(considered) >= self.parallel_min_branches and root_recursive_plies > 0:
            args = [
                {
                    "fen": board.fen(),
                    "record": record,
                    "selected": selected_uci,
                    "remaining_plies": root_recursive_plies,
                    "cdepth": self.cdepth,
                    "level": 0,
                    "style": self.style,
                    "beta": self.beta,
                    "deepening_quantile": self.deepening_quantile,
                    "search_mode": self.search_mode,
                    "eval_weights": self.evaluator.weights,
                    "parallel_min_branches": self.parallel_min_branches,
                    "kappa": self.kappa,
                    "node_shallow_value": current_u,
                    "current_u": current_u,
                    "current_branches": current_branches,
                    "detailed": record.uci in detailed_uci,
                }
                for record in considered
            ]
            self.diagnostics.parallel_branches += len(considered)
            with ProcessPoolExecutor(
                max_workers=self.search_workers,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                worker_results = list(executor.map(_branch_worker, args))
            for move, branch, child_diag in worker_results:
                moves_by_uci[move.uci] = move
                branch_by_uci[branch.uci] = branch
                child = SearchDiagnostics(
                    self.cdepth,
                    self.requested_recursive_plies,
                    self.deepening_quantile,
                    self.search_mode,
                    self.search_workers,
                    self.parallel_min_branches,
                )
                for diag_key, value in child_diag.items():
                    if diag_key in {"average_k", "average_response_k"}:
                        continue
                    if diag_key in child.__dataclass_fields__:
                        setattr(child, diag_key, value)
                self.diagnostics.add_child(child)
        else:
            for record in considered:
                branch_search = self._new_branch_context()
                move, branch = branch_search._evaluate_root_candidate_once(  # noqa: SLF001
                    board,
                    record,
                    selected,
                    root_recursive_plies,
                    0,
                    current_u,
                    current_u,
                    current_branches,
                    record.uci in detailed_uci,
                )
                before_clear = branch_search._cache_sizes()  # noqa: SLF001
                self.diagnostics.add_child(branch_search.diagnostics)
                branch_search.clear_branch_caches()
                self._branch_cache_lifecycle.append(
                    {"before_clear": before_clear, "after_clear": branch_search._cache_sizes()}  # noqa: SLF001
                )
                moves_by_uci[record.uci] = move
                branch_by_uci[record.uci] = branch

        moves = tuple(
            self._outside_consideration_move(record, current_u)
            if not record.in_consideration_set
            else moves_by_uci[record.uci]
            for record in landscape.records
        )
        root_branches = tuple(branch_by_uci[record.uci] for record in considered)
        root_value = side_to_move_backup(board.turn, root_branches) if root_branches else current_u
        root_key = self.position_key(board)
        root_phase = 0
        self._value_cache[(root_key, 0, root_phase)] = current_u
        self._selection_cache[(root_key, 0, root_phase)] = NodeSelection(
            entropy=landscape.entropy,
            effective_moves=landscape.effective_moves,
            expanded_count=0,
            selected_uci=(),
            branches=current_branches,
        )
        self._value_cache[(root_key, self.requested_recursive_plies, root_phase)] = root_value
        self._selection_cache[(root_key, self.requested_recursive_plies, root_phase)] = NodeSelection(
            entropy=landscape.entropy,
            effective_moves=landscape.effective_moves,
            expanded_count=expanded_count,
            selected_uci=selected_uci,
            branches=root_branches,
        )
        self._value_cache[(root_key, root_recursive_plies, root_phase)] = root_value
        self._selection_cache[(root_key, root_recursive_plies, root_phase)] = NodeSelection(
            entropy=landscape.entropy,
            effective_moves=landscape.effective_moves,
            expanded_count=expanded_count,
            selected_uci=selected_uci,
            branches=root_branches,
        )
        root_diagnostics = {
            **self.diagnostics.as_dict(),
            "configured_cdepth": self.cdepth,
            "configured_recursive_plies": self.requested_recursive_plies,
            "root_legal_move_count": landscape.legal_move_count,
            "root_consideration_count": landscape.consideration_count,
            "root_deepening_branch_count": expanded_count,
            "root_forced_continuation": root_forced_continuation,
            "root_adaptive_cycle_limit": root_cycle_limit,
            "root_adaptive_recursive_plies": root_recursive_plies,
            "legal_move_count": landscape.legal_move_count,
            "entropy": landscape.entropy,
            "n_eff": landscape.effective_moves,
            "consideration_count": landscape.consideration_count,
            "deepening_branch_count": expanded_count,
            "consideration_probability_mass_raw": landscape.consideration_probability_mass_raw,
            "deepening_quantile": self.deepening_quantile,
            "deepened_move_count": len(selected_uci),
            "deepening_probability_mass_considered": sum(
                record.probability for record in landscape.considered_records if record.uci in selected
            ),
        }
        result = SearchResult(
            board_fen=board.fen(),
            player=player,
            side=result_side,
            cdepth=self.cdepth,
            requested_recursive_plies=self.requested_recursive_plies,
            search_mode=self.search_mode,
            beta=self.beta,
            deepening_quantile=self.deepening_quantile,
            kappa=self.kappa,
            U=current_u,
            entropy=landscape.entropy,
            N_eff=landscape.effective_moves,
            K=landscape.consideration_count,
            selected_uci=selected_uci,
            moves=moves,
            diagnostics=root_diagnostics,
            evaluation_weights=self.evaluator.weights.as_dict(),
        )
        self._result_cache[key] = result
        return result

    def _candidate_thermodynamics(
        self,
        board: chess.Board,
        record: MoveRecord,
        branch: LandscapeObservation | None,
        current_u: float,
        current_branches: Tuple[LandscapeObservation, ...],
        *,
        detailed: bool,
    ) -> Dict[str, object]:
        from .thermodynamics import decompose_transition

        after = board.copy(stack=False)
        after.push(record.move)
        root_selected = branch.was_deepened if branch is not None else False
        remaining_depth = branch.remaining_plies if root_selected and branch is not None else 0
        recursive = self._recursive_result(after, remaining_depth, 1)
        terminal_u = recursive.value
        g_tilde = terminal_u - current_u
        principal_variation = (record.uci,) + recursive.principal_variation
        cycle_metadata = self._principal_variation_cycle_metadata(
            board,
            principal_variation,
            recursive.endpoint_fen,
        )
        search_endpoint_fen = str(cycle_metadata["search_endpoint_fen"])
        thermodynamic_endpoint_fen = cycle_metadata["thermodynamic_endpoint_fen"]
        recursive_plies_used = int(cycle_metadata["recursive_plies_used"])
        deepened_cycles = int(cycle_metadata["deepened_cycles"])

        q_tilde = None
        thermo_g_tilde = None
        w_tilde = None
        a_tilde = None
        decomposition_error = None
        qwa_unavailable_reason = None
        if deepened_cycles == 0 or thermodynamic_endpoint_fen is None:
            qwa_unavailable_reason = "no_completed_cycle"
        else:
            endpoint_board = chess.Board(str(thermodynamic_endpoint_fen))
            endpoint_u = self._future_subjective_value(endpoint_board, 0, 1)
            endpoint_branches = self._future_branch_observations(endpoint_board, 0, 1)
            dec = decompose_transition(current_branches, endpoint_branches)
            thermo_g_tilde = endpoint_u - current_u
            q_tilde = dec.delta_q
            w_tilde = dec.delta_w
            a_tilde = dec.delta_a
            decomposition_error = thermo_g_tilde - (q_tilde + w_tilde + a_tilde)

        if after.is_game_over(claim_draw=False):
            return {
                "terminal_u": terminal_u,
                "g_tilde": g_tilde,
                "retained_probability_mass": None,
                "retained_response_count": 0 if detailed else None,
                "total_probability_mass": None,
                "deepened_probability_mass": None,
                "nondeepened_probability_mass": None,
                "q_tilde": q_tilde if detailed else None,
                "thermo_g_tilde": thermo_g_tilde if detailed else None,
                "w_tilde": w_tilde if detailed else None,
                "a_tilde": a_tilde if detailed else None,
                "thermo_decomposition_error": decomposition_error if detailed else None,
                "response_entropy": None,
                "response_N_eff": None,
                "response_K": None,
                "responses": (),
                "principal_variation": principal_variation,
                "endpoint_fen": search_endpoint_fen,
                "search_endpoint_fen": search_endpoint_fen,
                "thermodynamic_endpoint_fen": thermodynamic_endpoint_fen,
                "recursive_plies_used": recursive_plies_used,
                "deepened_cycles": deepened_cycles,
                "qwa_unavailable_reason": qwa_unavailable_reason,
            }

        if not detailed:
            return {
                "terminal_u": terminal_u,
                "g_tilde": g_tilde,
                "retained_probability_mass": None,
                "retained_response_count": None,
                "total_probability_mass": None,
                "deepened_probability_mass": None,
                "nondeepened_probability_mass": None,
                "q_tilde": None,
                "thermo_g_tilde": None,
                "w_tilde": None,
                "a_tilde": None,
                "thermo_decomposition_error": None,
                "response_entropy": None,
                "response_N_eff": branch.response_neff if branch else None,
                "response_K": branch.response_k if branch else None,
                "responses": (),
                "principal_variation": principal_variation,
                "endpoint_fen": search_endpoint_fen,
                "search_endpoint_fen": search_endpoint_fen,
                "thermodynamic_endpoint_fen": thermodynamic_endpoint_fen,
                "recursive_plies_used": recursive_plies_used,
                "deepened_cycles": deepened_cycles,
                "qwa_unavailable_reason": qwa_unavailable_reason,
            }

        candidate_selection = self.node_selection(after, remaining_depth, 1)
        response_records = candidate_selection.branches if candidate_selection is not None else ()
        reply_records = {reply.uci: reply for reply in self.response_landscape(after).records}
        selected_rank = {
            uci: rank
            for rank, uci in enumerate(
                candidate_selection.selected_uci if candidate_selection is not None else (),
                start=1,
            )
        }
        selected_responses = tuple(response for response in response_records if response.was_deepened)
        selected_probability_mass = sum(response.probability for response in selected_responses)
        total_probability_mass = sum(response.probability for response in response_records)
        nondeepened_probability_mass = total_probability_mass - selected_probability_mass
        response_evaluations: list[ResponseEvaluation] = []
        selected_response_uci = recursive.principal_variation[0] if recursive.principal_variation else None
        for response in response_records:
            branch_g_tilde = response.observable_value - current_u
            reply_record = reply_records.get(response.uci)
            child = after.copy(stack=False)
            child.push(response.move)
            response_evaluations.append(
                ResponseEvaluation(
                    move=response.move,
                    uci=response.uci,
                    san=reply_record.san if reply_record is not None else response.uci,
                    probability=response.probability,
                    value=response.observable_value,
                    selected_for_refinement=response.was_deepened,
                    static_value=reply_record.static_after if reply_record is not None else None,
                    refinement_rank=selected_rank.get(response.uci),
                    depth_used=response.depth_used,
                    terminal_u=response.observable_value,
                    branch_g_tilde=branch_g_tilde,
                    branch_q_tilde=None,
                    branch_w_tilde=None,
                    branch_a_tilde=None,
                    branch_decomposition_error=None,
                    conditional_probability=response.probability,
                    used_shallow_landscape=not response.was_deepened,
                    static_components=reply_record.static_components_after if reply_record is not None else None,
                    board_fen=child.fen(),
                    static_terminal=reply_record.static_terminal_after if reply_record is not None else False,
                    static_terminal_reason=reply_record.static_terminal_reason_after if reply_record is not None else None,
                    selected_by_recursive_policy=response.uci == selected_response_uci,
                )
            )
        return {
            "terminal_u": terminal_u,
            "g_tilde": g_tilde,
            "q_tilde": q_tilde,
            "thermo_g_tilde": thermo_g_tilde,
            "w_tilde": w_tilde,
            "a_tilde": a_tilde,
            "retained_probability_mass": selected_probability_mass if detailed else None,
            "retained_response_count": len(selected_responses) if detailed else None,
            "total_probability_mass": total_probability_mass if detailed else None,
            "deepened_probability_mass": selected_probability_mass if detailed else None,
            "nondeepened_probability_mass": nondeepened_probability_mass if detailed else None,
            "thermo_decomposition_error": decomposition_error,
            "response_entropy": candidate_selection.entropy if candidate_selection is not None else None,
            "response_N_eff": candidate_selection.effective_moves if candidate_selection is not None else None,
            "response_K": candidate_selection.expanded_count if candidate_selection is not None else None,
            "responses": tuple(response_evaluations),
            "principal_variation": principal_variation,
            "endpoint_fen": search_endpoint_fen,
            "search_endpoint_fen": search_endpoint_fen,
            "thermodynamic_endpoint_fen": thermodynamic_endpoint_fen,
            "recursive_plies_used": recursive_plies_used,
            "deepened_cycles": deepened_cycles,
            "qwa_unavailable_reason": qwa_unavailable_reason,
        }

    def _expected_value(self, board: chess.Board, remaining_plies: int, level: int) -> float:
        self.diagnostics.maximum_depth_reached = max(
            self.diagnostics.maximum_depth_reached, level
        )
        position_key = self.position_key(board)
        phase = level % 2
        inherited_cache_key = (position_key, remaining_plies, phase)
        if inherited_cache_key in self._value_cache:
            self.diagnostics.cache_hits += 1
            return self._value_cache[inherited_cache_key]

        self.diagnostics.nodes_evaluated += 1
        if board.is_game_over(claim_draw=False):
            value = self.static_value(board)
            self._selection_cache[inherited_cache_key] = _terminal_observation(value, remaining_plies, self.search_mode)
            self._value_cache[inherited_cache_key] = value
            return value

        landscape = self.landscape(board)
        if not landscape.records:
            value = self.static_value(board)
            self._selection_cache[inherited_cache_key] = _terminal_observation(value, remaining_plies, self.search_mode)
            self._value_cache[inherited_cache_key] = value
            return value

        if remaining_plies == 0:
            branches = tuple(
                LandscapeObservation(
                    move=record.move,
                    uci=record.uci,
                    probability=record.probability,
                    observable_value=float(record.static_after),
                    was_deepened=False,
                    depth_used=0,
                    search_mode=self.search_mode,
                    remaining_plies=0,
                )
                for record in landscape.considered_records
            )
            selection = NodeSelection(
                entropy=landscape.entropy,
                effective_moves=landscape.effective_moves,
                expanded_count=0,
                selected_uci=(),
                branches=branches,
            )
            self._selection_cache[inherited_cache_key] = selection
            self._value_cache[inherited_cache_key] = landscape.expected_value
            return landscape.expected_value

        considered = tuple(record for record in landscape.records if record.in_consideration_set)
        ranked = ranked_for_refinement(considered, board.turn)
        expanded_count = deepening_count_for_quantile(
            (record.probability for record in ranked),
            self.deepening_quantile,
        )
        selected_uci = tuple(record.uci for record in ranked[:expanded_count])
        selected = set(selected_uci)
        effective_remaining, local_cycles, _ = self._effective_remaining_plies(
            expanded_count,
            remaining_plies,
            level,
        )
        self._record_adaptive_depth_decision(
            cycle_limit=local_cycles,
            inherited_remaining=remaining_plies,
            effective_remaining=effective_remaining,
            legal_move_count=landscape.legal_move_count,
            consideration_count=landscape.consideration_count,
            deepening_branch_count=expanded_count,
        )
        cache_key = (position_key, effective_remaining, phase)
        if cache_key in self._value_cache:
            self.diagnostics.cache_hits += 1
            value = self._value_cache[cache_key]
            if cache_key != inherited_cache_key:
                self._value_cache[inherited_cache_key] = value
                selection = self._selection_cache.get(cache_key)
                if selection is not None:
                    self._selection_cache[inherited_cache_key] = selection
            return value

        if effective_remaining == 0:
            branches = tuple(
                LandscapeObservation(
                    move=record.move,
                    uci=record.uci,
                    probability=record.probability,
                    observable_value=float(record.static_after),
                    was_deepened=False,
                    depth_used=0,
                    search_mode=self.search_mode,
                    remaining_plies=effective_remaining,
                )
                for record in landscape.considered_records
            )
            selection = NodeSelection(
                entropy=landscape.entropy,
                effective_moves=landscape.effective_moves,
                expanded_count=0,
                selected_uci=(),
                branches=branches,
            )
            self._selection_cache[cache_key] = selection
            self._selection_cache[inherited_cache_key] = selection
            self._value_cache[cache_key] = landscape.expected_value
            self._value_cache[inherited_cache_key] = landscape.expected_value
            return landscape.expected_value

        self.diagnostics.expanded_total += expanded_count
        self.diagnostics.expanded_nodes += 1

        branches = self._evaluate_branches(
            board,
            considered,
            selected,
            effective_remaining,
            level,
            landscape.expected_value,
        )
        value = side_to_move_backup(board.turn, branches)

        selection = NodeSelection(
            entropy=landscape.entropy,
            effective_moves=landscape.effective_moves,
            expanded_count=expanded_count,
            selected_uci=selected_uci,
            branches=tuple(branches),
        )
        self._selection_cache[cache_key] = selection
        self._selection_cache[inherited_cache_key] = selection
        self._value_cache[cache_key] = value
        self._value_cache[inherited_cache_key] = value
        return value

    def _evaluate_branches(
        self,
        board: chess.Board,
        records: Iterable[MoveRecord],
        selected: set[str],
        remaining_plies: int,
        level: int,
        node_shallow_value: float | None = None,
    ) -> Tuple[LandscapeObservation, ...]:
        records = tuple(records)
        should_parallel = (
            level == 0
            and self.search_workers > 1
            and len(records) >= self.parallel_min_branches
            and remaining_plies > 0
        )
        if not should_parallel and level == 0 and remaining_plies > 0:
            branches: list[LandscapeObservation] = []
            for record in records:
                branch_search = self._new_branch_context()
                branch = branch_search._evaluate_branch(  # noqa: SLF001
                    board,
                    record,
                    selected,
                    remaining_plies,
                    level,
                    node_shallow_value,
                )
                before_clear = branch_search._cache_sizes()  # noqa: SLF001
                child = branch_search.diagnostics
                self.diagnostics.add_child(child)
                branch_search.clear_branch_caches()
                self._branch_cache_lifecycle.append(
                    {"before_clear": before_clear, "after_clear": branch_search._cache_sizes()}  # noqa: SLF001
                )
                branches.append(branch)
            return tuple(branches)

        if not should_parallel:
            return tuple(
                self._evaluate_branch(board, record, selected, remaining_plies, level, node_shallow_value)
                for record in records
            )

        args = [
            {
                "fen": board.fen(),
                "record": record,
                "selected": tuple(selected),
                "remaining_plies": remaining_plies,
                "cdepth": self.cdepth,
                "level": level,
                "style": self.style,
                "beta": self.beta,
                "deepening_quantile": self.deepening_quantile,
                "search_mode": self.search_mode,
                "eval_weights": self.evaluator.weights,
                "parallel_min_branches": self.parallel_min_branches,
                "kappa": self.kappa,
                "node_shallow_value": node_shallow_value,
            }
            for record in records
        ]
        self.diagnostics.parallel_branches += len(records)
        with ProcessPoolExecutor(
            max_workers=self.search_workers,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            results = list(executor.map(_branch_observation_worker, args))
        branches: list[LandscapeObservation] = []
        for branch, child_diag in results:
            branches.append(branch)
            child = SearchDiagnostics(
                self.cdepth,
                self.requested_recursive_plies,
                self.deepening_quantile,
                self.search_mode,
                self.search_workers,
                self.parallel_min_branches,
            )
            for key, value in child_diag.items():
                if key in {"average_k", "average_response_k"}:
                    continue
                if key in child.__dataclass_fields__:
                    setattr(child, key, value)
            self.diagnostics.add_child(child)
        return tuple(branches)

    def _evaluate_branch(
        self,
        board: chess.Board,
        record: MoveRecord,
        selected: set[str],
        remaining_plies: int,
        level: int,
        node_shallow_value: float | None = None,
    ) -> LandscapeObservation:
        own_selected = record.uci in selected
        branch_value = float(record.static_after)
        response_neff = None
        response_k = None
        response_branches: list[ResponseObservation] = []

        after = board.copy(stack=False)
        after.push(record.move)
        if after.is_game_over(claim_draw=False):
            branch_value = self.static_value(after)
        else:
            remaining_after_move = remaining_plies - 1 if own_selected else 0
            branch_value = self._expected_value(after, remaining_after_move, level + 1)
            if own_selected:
                reply_selection = self.node_selection(after, remaining_after_move, level + 1)
                reply_landscape = self.response_landscape(after)
                response_neff = reply_landscape.effective_moves
                response_k = reply_selection.expanded_count if reply_selection is not None else 0
                if remaining_after_move > 0:
                    self.diagnostics.response_expanded_total += response_k
                    self.diagnostics.response_expanded_nodes += 1
                selected_reply_rank = {
                    uci: rank for rank, uci in enumerate(
                        reply_selection.selected_uci if reply_selection is not None else (),
                        start=1,
                    )
                }
                response_by_uci = {
                    response.uci: response
                    for response in (reply_selection.branches if reply_selection is not None else ())
                }
                for reply in reply_landscape.considered_records:
                    child = after.copy(stack=False)
                    child.push(reply.move)
                    response = response_by_uci.get(reply.uci)
                    reply_selected = reply.uci in selected_reply_rank
                    reply_value = (
                        response.observable_value
                        if response is not None
                        else self._future_subjective_value(child, 0, level + 1)
                    )
                    child_depth = remaining_after_move - 1 if reply_selected else 0
                    shallow_value = node_shallow_value
                    if shallow_value is None:
                        shallow_value = self._future_subjective_value(board, 0, 0)
                    du = reply_value - shallow_value
                    response_branches.append(
                        ResponseObservation(
                            move=reply.move,
                            uci=reply.uci,
                            san=after.san(reply.move),
                            probability=reply.probability,
                            response_value=reply_value,
                            selected_for_refinement=reply_selected,
                            depth_used=max(0, child_depth),
                            static_value=reply.static_after,
                            refinement_rank=selected_reply_rank.get(reply.uci),
                            used_shallow_landscape=not reply_selected,
                            conditional_probability=reply.probability,
                            branch_g_tilde=du,
                            branch_q_tilde=None,
                            branch_w_tilde=None,
                            branch_a_tilde=None,
                            branch_decomposition_error=None,
                            static_components=reply.static_components_after,
                            board_fen=child.fen(),
                            static_terminal=reply.static_terminal_after,
                            static_terminal_reason=reply.static_terminal_reason_after,
                        )
                    )
        return LandscapeObservation(
            move=record.move,
            uci=record.uci,
            probability=record.probability,
            observable_value=branch_value,
            was_deepened=own_selected,
            depth_used=remaining_plies if own_selected else 0,
            response_neff=response_neff,
            response_k=response_k,
            total_probability_mass=sum(response.probability for response in response_branches) if response_branches else None,
            deepened_probability_mass=sum(response.probability for response in response_branches if response.selected_for_refinement) if response_branches else None,
            nondeepened_probability_mass=sum(response.probability for response in response_branches if not response.selected_for_refinement) if response_branches else None,
            response_branches=tuple(response_branches),
            search_mode=self.search_mode,
            remaining_plies=remaining_plies,
        )
