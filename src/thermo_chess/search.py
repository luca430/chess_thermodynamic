"""Recursive bounded search over a fixed odd ply depth with adaptive breadth."""

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
from .measure import KAPPA, MoveLandscape, MoveRecord, Style, move_distribution


PositionKey = Tuple[str, bool, bool, bool]
SearchMode = Literal["accurate"]
def validate_depth(depth: int) -> int:
    if depth < 3:
        raise ValueError("depth must be an odd ply count at least 3")
    if depth % 2 == 0:
        raise ValueError("depth must be odd")
    return depth


def adaptive_breadth(effective_moves: float, adaptive_c: float, legal_moves: int) -> int:
    """Return the entropy-adaptive number of branches to refine."""
    if adaptive_c <= 0.0:
        raise ValueError("adaptive_c must be greater than 0")
    if legal_moves <= 0:
        return 0
    return min(legal_moves, max(1, math.ceil(adaptive_c * effective_moves)))


def ranked_for_refinement(
    records: Iterable[MoveRecord],
    side_to_move: chess.Color | None = None,
) -> Tuple[MoveRecord, ...]:
    """Rank retained branches by original transition probability."""
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

    def as_dict(self) -> Dict[str, object]:
        return {
            "uci": self.uci,
            "san": self.san,
            "probability": self.probability,
            "response_value": self.response_value,
            "value": self.response_value,
            "selected_for_refinement": self.selected_for_refinement,
            "depth_used": self.depth_used,
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
            depth_used=int(data.get("depth_used", 0)),
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
    response_branches: Tuple[ResponseObservation, ...] = ()
    search_mode: SearchMode = "accurate"
    depth: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "uci": self.uci,
            "probability": self.probability,
            "observable_value": self.observable_value,
            "branch_value": self.observable_value,
            "was_deepened": self.was_deepened,
            "selected_for_refinement": self.was_deepened,
            "actually_deepened": self.depth_used > 0,
            "depth_used": self.depth_used,
            "depth": self.depth,
            "search_mode": self.search_mode,
            "response_neff": self.response_neff,
            "response_k": self.response_k,
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
    static_value_after_move: float
    base_potential: float
    phase_potential: float
    features: Dict[str, float]
    static_components_after_move: Dict[str, float] | None = None
    static_terminal_after_move: bool = False
    static_terminal_reason_after_move: str | None = None
    terminal_u: float | None = None
    q_tilde: float | None = None
    w_tilde: float | None = None
    a_tilde: float | None = None
    retained_probability_mass: float | None = None
    retained_response_count: int | None = None
    thermo_decomposition_error: float | None = None
    response_entropy: float | None = None
    response_N_eff: float | None = None
    response_K: int | None = None
    responses: Tuple[ResponseEvaluation, ...] = ()
    depth_used: int = 0

    def as_branch_observation(self, *, depth: int, search_mode: SearchMode) -> LandscapeObservation:
        return LandscapeObservation(
            move=self.move,
            uci=self.uci,
            probability=self.probability,
            observable_value=self.branch_value,
            was_deepened=self.selected_for_refinement,
            depth_used=self.depth_used,
            response_neff=self.response_N_eff,
            response_k=self.response_K,
            response_branches=tuple(response.as_observation() for response in self.responses),
            search_mode=search_mode,
            depth=depth,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "san": self.san,
            "uci": self.uci,
            "probability": self.probability,
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
            "w_tilde": self.w_tilde,
            "a_tilde": self.a_tilde,
            "retained_probability_mass": self.retained_probability_mass,
            "retained_response_count": self.retained_response_count,
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
            "depth_used": self.depth_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "MoveEvaluation":
        uci = str(data["uci"])
        return cls(
            move=_move_from_uci(uci),
            san=str(data.get("san", "")),
            uci=uci,
            probability=float(data["probability"]),
            potential=float(data.get("potential", data.get("phi", 0.0))),
            branch_value=float(data.get("branch_value", 0.0)),
            g_tilde=None if data.get("g_tilde") is None else float(data["g_tilde"]),
            selected_for_refinement=bool(data.get("selected_for_refinement", False)),
            static_value_after_move=float(data.get("static_value_after_move", data.get("static_after", 0.0))),
            base_potential=float(data.get("base_potential", 0.0)),
            phase_potential=float(data.get("phase_potential", 0.0)),
            features=dict(data.get("features", {})),
            static_components_after_move=None if data.get("static_components_after_move") is None else {str(key): float(value) for key, value in dict(data["static_components_after_move"]).items()},
            static_terminal_after_move=bool(data.get("static_terminal_after_move", False)),
            static_terminal_reason_after_move=None if data.get("static_terminal_reason_after_move") is None else str(data["static_terminal_reason_after_move"]),
            terminal_u=None if data.get("terminal_u") is None else float(data["terminal_u"]),
            q_tilde=None if data.get("q_tilde") is None else float(data["q_tilde"]),
            w_tilde=None if data.get("w_tilde") is None else float(data["w_tilde"]),
            a_tilde=None if data.get("a_tilde") is None else float(data["a_tilde"]),
            retained_probability_mass=None if data.get("retained_probability_mass") is None else float(data["retained_probability_mass"]),
            retained_response_count=None if data.get("retained_response_count") is None else int(data["retained_response_count"]),
            thermo_decomposition_error=None if data.get("thermo_decomposition_error") is None else float(data["thermo_decomposition_error"]),
            response_entropy=None if data.get("response_entropy") is None else float(data["response_entropy"]),
            response_N_eff=None if data.get("response_N_eff") is None else float(data["response_N_eff"]),
            response_K=None if data.get("response_K") is None else int(data["response_K"]),
            responses=tuple(ResponseEvaluation.from_dict(item) for item in data.get("responses", [])),
            depth_used=int(data.get("depth_used", 0)),
        )


@dataclass(frozen=True)
class SearchResult:
    board_fen: str
    player: str
    side: str
    depth: int
    search_mode: SearchMode
    beta: float
    adaptive_c: float
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
            move.as_branch_observation(depth=self.depth, search_mode=self.search_mode)
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
            "depth": self.depth,
            "search_mode": self.search_mode,
            "adaptive_c": self.adaptive_c,
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
            "depth": self.depth,
            "search_mode": self.search_mode,
            "beta": self.beta,
            "kappa": self.kappa,
            "temperature": 1.0 / (self.kappa * self.beta),
            "adaptive_c": self.adaptive_c,
            "U": self.U,
            "entropy": self.entropy,
            "N_eff": self.N_eff,
            "K": self.K,
            "selected_uci": list(self.selected_uci),
            "moves": [move.as_dict() for move in self.moves],
            "diagnostics": self.diagnostics,
            "evaluation_weights": self.evaluation_weights,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SearchResult":
        return cls(
            board_fen=str(data["board_fen"]),
            player=str(data.get("player", "")),
            side=str(data.get("side", "")),
            depth=int(data.get("depth", 0)),
            search_mode=_validate_search_mode(str(data.get("search_mode", "accurate"))),
            beta=float(data.get("beta", 1.0)),
            kappa=float(data.get("kappa", KAPPA)),
            adaptive_c=float(data.get("adaptive_c", 0.3)),
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


@dataclass
class SearchDiagnostics:
    requested_depth: int
    adaptive_c: float
    search_mode: SearchMode = "accurate"
    search_workers: int = 1
    parallel_min_branches: int = 0
    nodes_evaluated: int = 0
    static_evaluations: int = 0
    recursive_nodes: int = 0
    maximum_depth_reached: int = 0
    cache_hits: int = 0
    expanded_total: int = 0
    expanded_nodes: int = 0
    response_expanded_total: int = 0
    response_expanded_nodes: int = 0
    parallel_branches: int = 0
    search_result_reuse_count: int = 0
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
        self.recursive_nodes += child.recursive_nodes
        self.maximum_depth_reached = max(self.maximum_depth_reached, child.maximum_depth_reached)
        self.cache_hits += child.cache_hits
        self.expanded_total += child.expanded_total
        self.expanded_nodes += child.expanded_nodes
        self.response_expanded_total += child.response_expanded_total
        self.response_expanded_nodes += child.response_expanded_nodes
        self.parallel_branches += child.parallel_branches
        self.search_result_reuse_count += child.search_result_reuse_count

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

    def components(
        self, board: chess.Board, context: BoardFeatureContext | None = None
    ) -> Dict[str, float] | None:
        return self.search.evaluator.components(board, context=context)

    def terminal_reason(self, board: chess.Board) -> str | None:
        return self.search.evaluator.terminal_reason(board)


def _terminal_observation(value: float, depth: int, search_mode: SearchMode) -> NodeSelection:
    branch = LandscapeObservation(
        move=chess.Move.null(),
        uci="__terminal__",
        probability=1.0,
        observable_value=value,
        was_deepened=False,
        depth_used=0,
        search_mode=search_mode,
        depth=depth,
    )
    return NodeSelection(0.0, 1.0, 0, (), (branch,))


def _branch_worker(args: Dict[str, object]) -> tuple[LandscapeObservation, Dict[str, float | int | str]]:
    evaluator = StaticEvaluator(args["eval_weights"])  # type: ignore[arg-type]
    search = AdaptiveExpectedValue(
        args["style"],  # type: ignore[arg-type]
        float(args["beta"]),
        evaluator,
        depth=int(args["depth"]),
        adaptive_c=float(args["adaptive_c"]),
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
        int(args["depth"]),
        int(args["level"]),
    )
    return branch, search.finish_diagnostics()


class AdaptiveExpectedValue:
    """Compute bounded expectations over a fixed odd ply depth with adaptive breadth."""

    def __init__(
        self,
        style: Style,
        beta: float,
        evaluator: StaticEvaluator,
        depth: int = 3,
        adaptive_c: float = 0.3,
        search_mode: SearchMode = "accurate",
        search_workers: int | None = 1,
        parallel_min_branches: int = 8,
        kappa: float = KAPPA,
    ) -> None:
        depth = validate_depth(depth)
        if adaptive_c <= 0.0:
            raise ValueError("adaptive_c must be greater than 0")
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
        self.depth = depth
        self.adaptive_c = adaptive_c
        self.search_mode = _validate_search_mode(search_mode)
        self.search_workers = search_workers
        self.parallel_min_branches = parallel_min_branches
        self._static_cache: Dict[PositionKey, float] = {}
        self._context_cache: Dict[PositionKey, BoardFeatureContext] = {}
        self._landscape_cache: Dict[PositionKey, MoveLandscape] = {}
        self._response_landscape_cache: Dict[PositionKey, MoveLandscape] = {}
        self._value_cache: Dict[Tuple[PositionKey, int], float] = {}
        self._selection_cache: Dict[Tuple[PositionKey, int], NodeSelection] = {}
        self._future_value_cache: Dict[Tuple[PositionKey, int], float] = {}
        self._future_selection_cache: Dict[Tuple[PositionKey, int], NodeSelection] = {}
        self._result_cache: Dict[Tuple[PositionKey, int, str], SearchResult] = {}
        self._cached_evaluator = _CachedEvaluator(self)
        self.diagnostics = SearchDiagnostics(
            depth,
            adaptive_c,
            self.search_mode,
            self.search_workers,
            self.parallel_min_branches,
        )
        self._started_at = 0.0

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
            self.depth,
            self.adaptive_c,
            self.search_mode,
            self.search_workers,
            self.parallel_min_branches,
        )
        self._started_at = time.perf_counter()

    def finish_diagnostics(self) -> Dict[str, float | int | str]:
        self.diagnostics.elapsed_time = time.perf_counter() - self._started_at
        return self.diagnostics.as_dict()

    def board_context(
        self, board: chess.Board, key: PositionKey | None = None
    ) -> BoardFeatureContext:
        position_key = key or self.position_key(board)
        cached = self._context_cache.get(position_key)
        if cached is not None:
            self.diagnostics.cache_hits += 1
            return cached
        context = board_context(board)
        self._context_cache[position_key] = context
        return context

    def static_value(
        self,
        board: chess.Board,
        key: PositionKey | None = None,
        context: BoardFeatureContext | None = None,
    ) -> float:
        position_key = key or self.position_key(board)
        if position_key in self._static_cache:
            self.diagnostics.cache_hits += 1
            return self._static_cache[position_key]
        values = context or self.board_context(board, position_key)
        value = self.evaluator.evaluate(board, context=values)
        self._static_cache[position_key] = value
        self.diagnostics.static_evaluations += 1
        return value

    def landscape(
        self, board: chess.Board, key: PositionKey | None = None
    ) -> MoveLandscape:
        position_key = key or self.position_key(board)
        if position_key in self._landscape_cache:
            self.diagnostics.cache_hits += 1
            return self._landscape_cache[position_key]
        context = self.board_context(board, position_key)
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

    def node_selection(self, board: chess.Board, depth: int | None = None) -> NodeSelection | None:
        remaining = self.depth if depth is None else depth
        return self._selection_cache.get((self.position_key(board), remaining))

    def branch_observations(
        self, board: chess.Board, depth: int | None = None
    ) -> Tuple[LandscapeObservation, ...]:
        remaining = self.depth if depth is None else depth
        self.expected_value(board, remaining)
        selection = self.node_selection(board, remaining)
        return selection.branches if selection is not None else ()

    def expected_value(self, board: chess.Board, depth: int | None = None) -> float:
        remaining = self.depth if depth is None else depth
        if remaining < 0:
            raise ValueError("depth must be at least 0")
        return self._expected_value(board, remaining, 0)

    def _future_subjective_value(self, board: chess.Board, remaining_depth: int, level: int) -> float:
        """Value at a same-player future state after a move-response pair was consumed.

        Public ``expected_value(..., 0)`` intentionally remains the static board
        value ``E(B)``. Candidate move scoring has different depth-zero
        semantics: the observer is to move again, so depth zero is the
        probability-weighted value of its immediate legal moves.
        """

        if remaining_depth < 0:
            raise ValueError("remaining_depth must be at least 0")
        if remaining_depth > 0:
            return self._expected_value(board, remaining_depth, level)

        self.diagnostics.maximum_depth_reached = max(
            self.diagnostics.maximum_depth_reached, level
        )
        position_key = self.position_key(board)
        cache_key = (position_key, remaining_depth)
        if cache_key in self._future_value_cache:
            self.diagnostics.cache_hits += 1
            return self._future_value_cache[cache_key]

        self.diagnostics.nodes_evaluated += 1
        if board.is_game_over(claim_draw=False):
            value = self.static_value(board)
            self._future_selection_cache[cache_key] = _terminal_observation(
                value, remaining_depth, self.search_mode
            )
            self._future_value_cache[cache_key] = value
            return value

        landscape = self.landscape(board)
        if not landscape.records:
            value = self.static_value(board)
            self._future_selection_cache[cache_key] = _terminal_observation(
                value, remaining_depth, self.search_mode
            )
            self._future_value_cache[cache_key] = value
            return value

        branches = tuple(
            LandscapeObservation(
                move=record.move,
                uci=record.uci,
                probability=record.probability,
                observable_value=record.static_after,
                was_deepened=False,
                depth_used=0,
                search_mode=self.search_mode,
                depth=remaining_depth,
            )
            for record in landscape.records
        )
        self._future_selection_cache[cache_key] = NodeSelection(
            entropy=landscape.entropy,
            effective_moves=landscape.effective_moves,
            expanded_count=0,
            selected_uci=(),
            branches=branches,
        )
        self._future_value_cache[cache_key] = landscape.expected_value
        return landscape.expected_value

    def _future_branch_observations(
        self, board: chess.Board, remaining_depth: int, level: int
    ) -> Tuple[LandscapeObservation, ...]:
        if remaining_depth > 0:
            self._expected_value(board, remaining_depth, level)
            selection = self.node_selection(board, remaining_depth)
        else:
            self._future_subjective_value(board, remaining_depth, level)
            selection = self._future_selection_cache.get(
                (self.position_key(board), remaining_depth)
            )
        return selection.branches if selection is not None else ()

    def search_result(self, board: chess.Board, player: str = "", side: str | None = None) -> SearchResult:
        result_side = side or ("white" if board.turn == chess.WHITE else "black")
        key = (self.position_key(board), self.depth, result_side)
        cached = self._result_cache.get(key)
        if cached is not None:
            self.diagnostics.search_result_reuse_count += 1
            return cached
        self.expected_value(board, self.depth)
        selection = self.node_selection(board, self.depth)
        landscape = self.landscape(board)
        if selection is None:
            moves: Tuple[MoveEvaluation, ...] = ()
            selected_uci: Tuple[str, ...] = ()
            expanded_count = 0
        else:
            branch_by_uci = {branch.uci: branch for branch in selection.branches}
            current_u = self._future_subjective_value(board, 0, 0)
            current_branches = self._future_branch_observations(board, 0, 0)
            self._qwa_root_board = board.copy(stack=False)
            fast_thermos = {}
            for record in landscape.records:
                branch = branch_by_uci.get(record.uci)
                fast_thermos[record.uci] = self._candidate_thermodynamics(
                    board, record, branch, current_u, current_branches, detailed=False
                )

            refined_records = [record for record in landscape.records if record.uci in selection.selected_uci]
            if not refined_records:
                moves = ()
                selected_uci = selection.selected_uci
                expanded_count = selection.expanded_count
                self._qwa_root_board = None
                return SearchResult(
                    board_fen=board.fen(),
                    player=player,
                    side=result_side,
                    depth=self.depth,
                    search_mode=self.search_mode,
                    beta=self.beta,
                    adaptive_c=self.adaptive_c,
                    kappa=self.kappa,
                    U=self._future_subjective_value(board, 0, 0),
                    entropy=landscape.entropy,
                    N_eff=landscape.effective_moves,
                    K=expanded_count,
                    selected_uci=selected_uci,
                    moves=moves,
                    diagnostics=self.diagnostics.as_dict(),
                    evaluation_weights=self.evaluator.weights.as_dict(),
                )
            detailed_uci = set(selection.selected_uci)

            moves_list = []
            for record in landscape.records:
                branch = branch_by_uci.get(record.uci)
                branch_value = branch.observable_value if branch else record.static_after
                thermo = fast_thermos[record.uci]
                if record.uci in detailed_uci:
                    detailed_thermo = self._candidate_thermodynamics(
                        board, record, branch, current_u, current_branches, detailed=True
                    )
                    thermo = {**thermo, **detailed_thermo}
                    thermo["terminal_u"] = fast_thermos[record.uci]["terminal_u"]
                    thermo["g_tilde"] = fast_thermos[record.uci]["g_tilde"]
                moves_list.append(
                    MoveEvaluation(
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
                        static_components_after_move=record.static_components_after,
                        static_terminal_after_move=record.static_terminal_after,
                        static_terminal_reason_after_move=record.static_terminal_reason_after,
                        terminal_u=thermo["terminal_u"],
                        g_tilde=thermo["g_tilde"],
                        q_tilde=thermo["q_tilde"],
                        w_tilde=thermo["w_tilde"],
                        a_tilde=thermo["a_tilde"],
                        retained_probability_mass=thermo["retained_probability_mass"],
                        retained_response_count=thermo["retained_response_count"],
                        thermo_decomposition_error=thermo["thermo_decomposition_error"],
                        response_entropy=thermo["response_entropy"],
                        response_N_eff=thermo["response_N_eff"],
                        response_K=thermo["response_K"],
                        responses=thermo["responses"],
                        depth_used=branch.depth_used if branch else 0,
                    )
                )
            moves = tuple(moves_list)
            selected_uci = selection.selected_uci
            expanded_count = selection.expanded_count
            self._qwa_root_board = None
        result = SearchResult(
            board_fen=board.fen(),
            player=player,
            side=result_side,
            depth=self.depth,
            search_mode=self.search_mode,
            beta=self.beta,
            adaptive_c=self.adaptive_c,
            kappa=self.kappa,
            U=self._future_subjective_value(board, 0, 0),
            entropy=landscape.entropy,
            N_eff=landscape.effective_moves,
            K=expanded_count,
            selected_uci=selected_uci,
            moves=moves,
            diagnostics=self.diagnostics.as_dict(),
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

        branch_value = branch.observable_value if branch else record.static_after
        after = board.copy(stack=False)
        after.push(record.move)
        if after.is_game_over(claim_draw=False):
            delta = record.static_after - current_u
            return {
                "terminal_u": record.static_after,
                "g_tilde": delta,
                "retained_probability_mass": None,
                "retained_response_count": 0 if detailed else None,
                "q_tilde": 0.0 if detailed else None,
                "w_tilde": 0.0 if detailed else None,
                "a_tilde": delta if detailed else None,
                "thermo_decomposition_error": 0.0 if detailed else None,
                "response_entropy": None,
                "response_N_eff": None,
                "response_K": None,
                "responses": (),
            }

        if (
            branch is None
            or not branch.was_deepened
            or not branch.response_branches
        ):
            return {
                "terminal_u": None,
                "g_tilde": None,
                "retained_probability_mass": None,
                "retained_response_count": None,
                "q_tilde": None,
                "w_tilde": None,
                "a_tilde": None,
                "thermo_decomposition_error": None,
                "response_entropy": None,
                "response_N_eff": branch.response_neff if branch else None,
                "response_K": branch.response_k if branch else None,
                "responses": (),
            }

        response_records = branch.response_branches
        selected_responses = tuple(
            response for response in response_records if response.selected_for_refinement
        )
        selected_probability_mass = sum(response.probability for response in selected_responses)

        response_evaluations: list[ResponseEvaluation] = []
        expected_next_u = 0.0
        g_tilde = None
        q_tilde = None
        w_tilde = None
        a_tilde = None
        decomposition_error = None
        for response in response_records:
            next_u = response.response_value
            conditional = response.conditional_probability if response.conditional_probability is not None else 0.0
            expected_next_u += conditional * next_u

            if detailed and response.selected_for_refinement:
                child = after.copy(stack=False)
                child.push(response.move)
                paths = self._terminal_paths(child, max(0, response.depth_used), 0)
                branch_value = sum(path[0] * path[1] for path in paths)
                branch_g_tilde = branch_value - current_u
                branch_q_tilde = sum(path[0] * path[2] for path in paths)
                branch_w_tilde = sum(path[0] * path[3] for path in paths)
                branch_a_tilde = sum(path[0] * path[4] for path in paths)
                branch_error = branch_g_tilde - (
                    branch_q_tilde + branch_w_tilde + branch_a_tilde
                )
                response_evaluations.append(
                    ResponseEvaluation(
                        move=response.move,
                        uci=response.uci,
                        san=response.san,
                        probability=response.probability,
                        value=branch_value,
                        selected_for_refinement=True,
                        static_value=response.static_value,
                        refinement_rank=response.refinement_rank,
                        depth_used=response.depth_used,
                        terminal_u=branch_value,
                        branch_g_tilde=branch_g_tilde,
                        branch_q_tilde=branch_q_tilde,
                        branch_w_tilde=branch_w_tilde,
                        branch_a_tilde=branch_a_tilde,
                        branch_decomposition_error=branch_error,
                        conditional_probability=conditional,
                        used_shallow_landscape=response.used_shallow_landscape,
                        static_components=response.static_components,
                        board_fen=response.board_fen,
                        static_terminal=response.static_terminal,
                        static_terminal_reason=response.static_terminal_reason,
                    )
                )

        g_tilde = expected_next_u - current_u
        if detailed and response_evaluations and selected_probability_mass > 0.0:
            g_tilde = sum(
                response.conditional_probability * response.branch_g_tilde
                for response in response_evaluations
                if response.conditional_probability is not None
                and response.branch_g_tilde is not None
            )
            q_tilde = sum(
                response.conditional_probability * response.branch_q_tilde
                for response in response_evaluations
                if response.conditional_probability is not None
                and response.branch_q_tilde is not None
            )
            w_tilde = sum(
                response.conditional_probability * response.branch_w_tilde
                for response in response_evaluations
                if response.conditional_probability is not None
                and response.branch_w_tilde is not None
            )
            a_tilde = sum(
                response.conditional_probability * response.branch_a_tilde
                for response in response_evaluations
                if response.conditional_probability is not None
                and response.branch_a_tilde is not None
            )
            decomposition_error = g_tilde - (
                q_tilde + w_tilde + a_tilde
            )
        return {
            "terminal_u": expected_next_u,
            "g_tilde": g_tilde,
            "q_tilde": q_tilde,
            "w_tilde": w_tilde,
            "a_tilde": a_tilde,
            "retained_probability_mass": selected_probability_mass if detailed else None,
            "retained_response_count": len(selected_responses) if detailed else None,
            "thermo_decomposition_error": decomposition_error,
            "response_entropy": None,
            "response_N_eff": branch.response_neff,
            "response_K": branch.response_k,
            "responses": tuple(response_evaluations),
        }

    def _terminal_paths(
        self, board: chess.Board, remaining_plies: int, level: int
    ) -> Tuple[Tuple[float, float, float, float, float], ...]:
        """Return retained terminal paths as (prob, Uhalf, Q, W, A)."""
        from .thermodynamics import decompose_transition

        if remaining_plies < 0:
            raise ValueError("remaining_plies must be non-negative")
        if remaining_plies == 0 or board.is_game_over(claim_draw=False):
            root_board = getattr(self, "_qwa_root_board", None)
            if root_board is None:
                value = self._future_subjective_value(board, 0, level)
                return ((1.0, value, 0.0, 0.0, value),)
            current_u = self._future_subjective_value(root_board, 0, 0)
            current_branches = self._future_branch_observations(root_board, 0, 0)
            future_u = self._future_subjective_value(board, 0, level)
            future_branches = self._future_branch_observations(board, 0, level)
            dec = decompose_transition(
                current_branches,
                future_branches,
            )
            delta_u = future_u - current_u
            delta_a = delta_u - dec.delta_q - dec.delta_w
            return ((1.0, future_u, dec.delta_q, dec.delta_w, delta_a),)

        landscape = self.landscape(board)
        if not landscape.records:
            return self._terminal_paths(board, 0, level)
        k = adaptive_breadth(landscape.effective_moves, self.adaptive_c, len(landscape.records))
        retained = ranked_for_refinement(landscape.records, board.turn)[:k]
        mass = sum(record.probability for record in retained)
        paths: list[Tuple[float, float, float, float, float]] = []
        for record in retained:
            child = board.copy(stack=False)
            child.push(record.move)
            conditional = record.probability / mass if mass > 0.0 else 0.0
            self.diagnostics.recursive_nodes += 1
            for prob, value, q, w, a in self._terminal_paths(
                child, remaining_plies - 1, level + 1
            ):
                paths.append((conditional * prob, value, q, w, a))
        return tuple(paths)

    def _expected_value(self, board: chess.Board, depth: int, level: int) -> float:
        self.diagnostics.maximum_depth_reached = max(
            self.diagnostics.maximum_depth_reached, level
        )
        position_key = self.position_key(board)
        cache_key = (position_key, depth)
        if cache_key in self._value_cache:
            self.diagnostics.cache_hits += 1
            return self._value_cache[cache_key]

        self.diagnostics.nodes_evaluated += 1
        if depth == 0:
            value = self.static_value(board)
            self._value_cache[cache_key] = value
            return value
        if board.is_game_over(claim_draw=False):
            value = self.static_value(board)
            self._selection_cache[cache_key] = _terminal_observation(value, depth, self.search_mode)
            self._value_cache[cache_key] = value
            return value

        landscape = self.landscape(board)
        if not landscape.records:
            value = self.static_value(board)
            self._selection_cache[cache_key] = _terminal_observation(value, depth, self.search_mode)
            self._value_cache[cache_key] = value
            return value

        expanded_count = adaptive_breadth(
            landscape.effective_moves, self.adaptive_c, len(landscape.records)
        )
        ranked = ranked_for_refinement(landscape.records, board.turn)
        selected_uci = tuple(record.uci for record in ranked[:expanded_count])
        selected = set(selected_uci)
        self.diagnostics.expanded_total += expanded_count
        self.diagnostics.expanded_nodes += 1

        branches = self._evaluate_branches(board, landscape.records, selected, depth, level)
        retained_mass = sum(branch.probability for branch in branches if branch.was_deepened)
        value = sum(
            (branch.probability / retained_mass) * branch.observable_value
            for branch in branches
            if branch.was_deepened and retained_mass > 0.0
        )

        self._selection_cache[cache_key] = NodeSelection(
            entropy=landscape.entropy,
            effective_moves=landscape.effective_moves,
            expanded_count=expanded_count,
            selected_uci=selected_uci,
            branches=tuple(branches),
        )
        self._value_cache[cache_key] = value
        return value

    def _evaluate_branches(
        self,
        board: chess.Board,
        records: Iterable[MoveRecord],
        selected: set[str],
        depth: int,
        level: int,
    ) -> Tuple[LandscapeObservation, ...]:
        records = tuple(records)
        should_parallel = (
            level == 0
            and self.search_workers > 1
            and len(records) >= self.parallel_min_branches
            and depth > 0
        )
        if not should_parallel:
            return tuple(
                self._evaluate_branch(board, record, selected, depth, level)
                for record in records
            )

        args = [
            {
                "fen": board.fen(),
                "record": record,
                "selected": tuple(selected),
                "depth": depth,
                "level": level,
                "style": self.style,
                "beta": self.beta,
                "adaptive_c": self.adaptive_c,
                "search_mode": self.search_mode,
                "eval_weights": self.evaluator.weights,
                "parallel_min_branches": self.parallel_min_branches,
                "kappa": self.kappa,
            }
            for record in records
        ]
        self.diagnostics.parallel_branches += len(records)
        with ProcessPoolExecutor(
            max_workers=self.search_workers,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            results = list(executor.map(_branch_worker, args))
        branches: list[LandscapeObservation] = []
        for branch, child_diag in results:
            branches.append(branch)
            child = SearchDiagnostics(
                self.depth,
                self.adaptive_c,
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
        depth: int,
        level: int,
    ) -> LandscapeObservation:
        own_selected = record.uci in selected
        branch_value = record.static_after
        response_neff = None
        response_k = None
        response_branches: list[ResponseObservation] = []

        after = board.copy(stack=False)
        after.push(record.move)
        if after.is_game_over(claim_draw=False):
            branch_value = self.static_value(after)
        else:
            if own_selected:
                reply_landscape = self.response_landscape(after)
                response_neff = reply_landscape.effective_moves
                response_k = adaptive_breadth(reply_landscape.effective_moves, self.adaptive_c, len(reply_landscape.records))
                reply_ranked = ranked_for_refinement(reply_landscape.records, after.turn)
                retained = tuple(reply_ranked[:response_k])
                retained_mass = sum(reply.probability for reply in retained)
                selected_reply_rank = {reply.uci: rank for rank, reply in enumerate(retained, start=1)}
                self.diagnostics.response_expanded_total += response_k
                self.diagnostics.response_expanded_nodes += 1
                branch_value = 0.0
                for reply in retained:
                    child = after.copy(stack=False)
                    child.push(reply.move)
                    conditional = reply.probability / retained_mass if retained_mass > 0.0 else 0.0
                    paths = self._terminal_paths(child, max(0, depth - 3), level + 1)
                    reply_value = sum(path[0] * path[1] for path in paths)
                    dq = sum(path[0] * path[2] for path in paths)
                    dw = sum(path[0] * path[3] for path in paths)
                    da = sum(path[0] * path[4] for path in paths)
                    du = reply_value - self._future_subjective_value(board, 0, 0)
                    err = du - (dq + dw + da)
                    branch_value += conditional * reply_value
                    response_branches.append(
                        ResponseObservation(
                            move=reply.move,
                            uci=reply.uci,
                            san=after.san(reply.move),
                            probability=reply.probability,
                            response_value=reply_value,
                            selected_for_refinement=True,
                            depth_used=max(0, depth - 3),
                            static_value=reply.static_after,
                            refinement_rank=selected_reply_rank.get(reply.uci),
                            used_shallow_landscape=depth == 3,
                            conditional_probability=conditional,
                            branch_g_tilde=du,
                            branch_q_tilde=dq,
                            branch_w_tilde=dw,
                            branch_a_tilde=da,
                            branch_decomposition_error=err,
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
            depth_used=depth if own_selected else 0,
            response_neff=response_neff,
            response_k=response_k,
            response_branches=tuple(response_branches),
            search_mode=self.search_mode,
            depth=depth,
        )
