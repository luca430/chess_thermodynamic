"""Entropy-adaptive recursive expected values over complete move-response cycles."""

from __future__ import annotations

import math
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
CandidateThermoMode = Literal["selected", "refined", "all"]
RefinementPolicy = Literal["static_eval", "probability"]


@dataclass(frozen=True)
class AdaptiveDepthThresholds:
    depth4_max_neff: float = 4.0
    depth3_max_neff: float = 8.0
    depth2_max_neff: float = 15.0

    def __post_init__(self) -> None:
        if not (
            0.0 < self.depth4_max_neff
            < self.depth3_max_neff
            < self.depth2_max_neff
        ):
            raise ValueError("adaptive depth thresholds must be positive and increasing")


def depth_from_effective_moves(
    effective_moves: float,
    thresholds: AdaptiveDepthThresholds | None = None,
) -> int:
    """Map effective move count to the entropy-selected cycle-depth cap."""
    limits = thresholds or AdaptiveDepthThresholds()
    if effective_moves <= limits.depth4_max_neff:
        return 4
    if effective_moves <= limits.depth3_max_neff:
        return 3
    if effective_moves <= limits.depth2_max_neff:
        return 2
    return 1


def local_search_depth(
    remaining_depth: int,
    effective_moves: float,
    thresholds: AdaptiveDepthThresholds | None = None,
) -> int:
    """Apply the entropy cap without increasing the remaining cycle budget."""
    if remaining_depth < 0:
        raise ValueError("remaining_depth must be at least 0")
    return min(
        remaining_depth,
        depth_from_effective_moves(effective_moves, thresholds),
    )


def adaptive_breadth(effective_moves: float, adaptive_c: float, legal_moves: int) -> int:
    """Return the entropy-adaptive number of branches to refine."""
    if adaptive_c <= 0.0:
        raise ValueError("adaptive_c must be greater than 0")
    if legal_moves <= 0:
        return 0
    return min(legal_moves, max(1, math.ceil(adaptive_c * effective_moves)))


def ranked_for_refinement(
    records: Iterable[MoveRecord],
    side_to_move: chess.Color,
    policy: RefinementPolicy = "static_eval",
) -> Tuple[MoveRecord, ...]:
    """Rank branches for deeper work.

    ``static_eval`` ranks by immediate board value in the side-to-move's favor.
    ``probability`` ranks by the move probability assigned by the observer's
    current policy distribution. This only allocates computation; probabilities
    still weight values.
    """

    if policy == "probability":
        return tuple(sorted(records, key=lambda record: record.probability, reverse=True))
    if policy == "static_eval":
        return tuple(
            sorted(
                records,
                key=lambda record: record.static_after,
                reverse=side_to_move == chess.WHITE,
            )
        )
    raise ValueError("refinement_policy must be 'static_eval' or 'probability'")


def _validate_search_mode(search_mode: str) -> SearchMode:
    if search_mode != "accurate":
        raise ValueError("search_mode must be 'accurate'")
    return search_mode  # type: ignore[return-value]


def _validate_refinement_policy(refinement_policy: str) -> RefinementPolicy:
    if refinement_policy not in {"static_eval", "probability"}:
        raise ValueError("refinement_policy must be 'static_eval' or 'probability'")
    return refinement_policy  # type: ignore[return-value]


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
        )


@dataclass(frozen=True)
class AdaptiveBranchObservation:
    move: chess.Move
    uci: str
    probability: float
    adaptive_branch_value: float
    was_deepened: bool
    depth_used: int
    response_neff: float | None = None
    response_k: int | None = None
    response_branches: Tuple[ResponseObservation, ...] = ()
    search_mode: SearchMode = "accurate"
    cdepth: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "uci": self.uci,
            "probability": self.probability,
            "adaptive_branch_value": self.adaptive_branch_value,
            "branch_value": self.adaptive_branch_value,
            "was_deepened": self.was_deepened,
            "selected_for_refinement": self.was_deepened,
            "actually_deepened": self.depth_used > 1,
            "depth_used": self.depth_used,
            "cdepth": self.cdepth,
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
    next_state_U: float | None = None
    delta_U: float | None = None
    delta_Q: float | None = None
    delta_W: float | None = None
    delta_A: float | None = None
    thermo_decomposition_error: float | None = None
    used_shallow_landscape: bool = False

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
        )

    def as_dict(self) -> Dict[str, object]:
        data = self.as_observation().as_dict()
        data.update({
            "next_state_U": self.next_state_U,
            "delta_U": self.delta_U,
            "delta_Q": self.delta_Q,
            "delta_W": self.delta_W,
            "delta_A": self.delta_A,
            "thermo_decomposition_error": self.thermo_decomposition_error,
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
            next_state_U=None if data.get("next_state_U") is None else float(data["next_state_U"]),
            delta_U=None if data.get("delta_U") is None else float(data["delta_U"]),
            delta_Q=None if data.get("delta_Q") is None else float(data["delta_Q"]),
            delta_W=None if data.get("delta_W") is None else float(data["delta_W"]),
            delta_A=None if data.get("delta_A") is None else float(data["delta_A"]),
            thermo_decomposition_error=None if data.get("thermo_decomposition_error") is None else float(data["thermo_decomposition_error"]),
            used_shallow_landscape=bool(data.get("used_shallow_landscape", False)),
        )


@dataclass(frozen=True)
class MoveEvaluation:
    move: chess.Move
    san: str
    uci: str
    probability: float
    potential: float
    branch_value: float
    candidate_delta_u: float
    selected_for_refinement: bool
    static_value_after_move: float
    base_potential: float
    phase_potential: float
    features: Dict[str, float]
    expected_next_U: float | None = None
    expected_delta_u_star: float | None = None
    candidate_delta_q: float | None = None
    candidate_delta_w: float | None = None
    candidate_delta_a: float | None = None
    candidate_delta_u_star: float | None = None
    shallow_expected_delta_u_star: float | None = None
    thermo_decomposition_error: float | None = None
    response_entropy: float | None = None
    response_N_eff: float | None = None
    response_K: int | None = None
    responses: Tuple[ResponseEvaluation, ...] = ()
    depth_used: int = 0

    def as_branch_observation(self, *, cdepth: int, search_mode: SearchMode) -> AdaptiveBranchObservation:
        return AdaptiveBranchObservation(
            move=self.move,
            uci=self.uci,
            probability=self.probability,
            adaptive_branch_value=self.branch_value,
            was_deepened=self.selected_for_refinement,
            depth_used=self.depth_used,
            response_neff=self.response_N_eff,
            response_k=self.response_K,
            response_branches=tuple(response.as_observation() for response in self.responses),
            search_mode=search_mode,
            cdepth=cdepth,
        )

    def as_dict(self) -> Dict[str, object]:
        expected_delta = self.expected_delta_u_star if self.expected_delta_u_star is not None else self.candidate_delta_u
        return {
            "san": self.san,
            "uci": self.uci,
            "probability": self.probability,
            "phi": self.potential,
            "potential": self.potential,
            "static_after": self.static_value_after_move,
            "static_value_after_move": self.static_value_after_move,
            "selection_value": self.expected_next_U if self.expected_next_U is not None else self.branch_value,
            "reply_expected_value": self.expected_next_U if self.expected_next_U is not None else self.branch_value,
            "branch_value": self.branch_value,
            "expected_next_U": self.expected_next_U,
            "expected_delta_u_star": expected_delta,
            "expected_delta_U_star": expected_delta,
            "candidate_delta_u": expected_delta,
            "candidate_delta_U": expected_delta,
            "delta_u": expected_delta,
            "candidate_delta_q": self.candidate_delta_q,
            "candidate_delta_Q": self.candidate_delta_q,
            "delta_q": self.candidate_delta_q,
            "candidate_delta_w": self.candidate_delta_w,
            "candidate_delta_W": self.candidate_delta_w,
            "delta_w": self.candidate_delta_w,
            "candidate_delta_a": self.candidate_delta_a,
            "candidate_delta_A": self.candidate_delta_a,
            "delta_a": self.candidate_delta_a,
            "candidate_delta_u_star": expected_delta,
            "candidate_delta_U_star": expected_delta,
            "delta_u_star": expected_delta,
            "shallow_expected_delta_u_star": self.shallow_expected_delta_u_star,
            "thermo_decomposition_error": self.thermo_decomposition_error,
            "selected_for_refinement": self.selected_for_refinement,
            "selected_for_deeper_analysis": self.selected_for_refinement,
            "actually_deepened": self.depth_used > 1,
            "base_potential": self.base_potential,
            "phase_potential": self.phase_potential,
            "total_potential": self.potential,
            "features": self.features,
            "response_entropy": self.response_entropy,
            "response_N_eff": self.response_N_eff,
            "response_neff": self.response_N_eff,
            "response_K": self.response_K,
            "response_k": self.response_K,
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
            branch_value=float(data.get("branch_value", data.get("selection_value", 0.0))),
            candidate_delta_u=float(data.get("candidate_delta_u", data.get("delta_u", 0.0))),
            selected_for_refinement=bool(data.get("selected_for_refinement", data.get("selected_for_deeper_analysis", False))),
            static_value_after_move=float(data.get("static_value_after_move", data.get("static_after", 0.0))),
            base_potential=float(data.get("base_potential", 0.0)),
            phase_potential=float(data.get("phase_potential", 0.0)),
            features=dict(data.get("features", {})),
            expected_next_U=None if data.get("expected_next_U") is None else float(data["expected_next_U"]),
            expected_delta_u_star=None if data.get("expected_delta_u_star", data.get("expected_delta_U_star")) is None else float(data.get("expected_delta_u_star", data.get("expected_delta_U_star"))),
            candidate_delta_q=None if data.get("candidate_delta_q", data.get("candidate_delta_Q", data.get("delta_q"))) is None else float(data.get("candidate_delta_q", data.get("candidate_delta_Q", data.get("delta_q")))),
            candidate_delta_w=None if data.get("candidate_delta_w", data.get("candidate_delta_W", data.get("delta_w"))) is None else float(data.get("candidate_delta_w", data.get("candidate_delta_W", data.get("delta_w")))),
            candidate_delta_a=None if data.get("candidate_delta_a", data.get("candidate_delta_A", data.get("delta_a"))) is None else float(data.get("candidate_delta_a", data.get("candidate_delta_A", data.get("delta_a")))),
            candidate_delta_u_star=None if data.get("candidate_delta_u_star", data.get("candidate_delta_U_star", data.get("delta_u_star"))) is None else float(data.get("candidate_delta_u_star", data.get("candidate_delta_U_star", data.get("delta_u_star")))),
            shallow_expected_delta_u_star=None if data.get("shallow_expected_delta_u_star") is None else float(data["shallow_expected_delta_u_star"]),
            thermo_decomposition_error=None if data.get("thermo_decomposition_error") is None else float(data["thermo_decomposition_error"]),
            response_entropy=None if data.get("response_entropy") is None else float(data["response_entropy"]),
            response_N_eff=None if data.get("response_N_eff", data.get("response_neff")) is None else float(data.get("response_N_eff", data.get("response_neff"))),
            response_K=None if data.get("response_K", data.get("response_k")) is None else int(data.get("response_K", data.get("response_k"))),
            responses=tuple(ResponseEvaluation.from_dict(item) for item in data.get("responses", [])),
            depth_used=int(data.get("depth_used", 0)),
        )


@dataclass(frozen=True)
class SearchResult:
    board_fen: str
    player: str
    side: str
    cdepth: int
    search_mode: SearchMode
    beta: float
    adaptive_c: float
    U: float
    entropy: float
    N_eff: float
    K: int
    selected_depth: int
    selected_uci: Tuple[str, ...]
    moves: Tuple[MoveEvaluation, ...]
    diagnostics: Dict[str, float | int | str]
    refinement_policy: RefinementPolicy = "static_eval"
    kappa: float = KAPPA

    def branch_observations(self) -> Tuple[AdaptiveBranchObservation, ...]:
        return tuple(
            move.as_branch_observation(cdepth=self.cdepth, search_mode=self.search_mode)
            for move in self.moves
        )

    def as_node_selection(self) -> "NodeSelection":
        return NodeSelection(
            entropy=self.entropy,
            effective_moves=self.N_eff,
            selected_depth=self.selected_depth,
            expanded_count=self.K,
            selected_uci=self.selected_uci,
            branches=self.branch_observations(),
        )

    def as_viewer_panel(self, board_turn: chess.Color) -> Dict[str, object]:
        moves = [move.as_dict() for move in self.moves]
        if self.refinement_policy == "probability":
            moves.sort(key=lambda item: item["probability"], reverse=True)
            sort_direction = "probability_descending"
        elif board_turn == chess.WHITE:
            moves.sort(key=lambda item: item["static_after"], reverse=True)
            sort_direction = "static_eval_descending"
        else:
            moves.sort(key=lambda item: item["static_after"])
            sort_direction = "static_eval_ascending"
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
            "selected_depth": self.selected_depth,
            "expanded_count": self.K,
            "cdepth": self.cdepth,
            "search_depth": self.cdepth,
            "search_mode": self.search_mode,
            "adaptive_c": self.adaptive_c,
            "kappa": self.kappa,
            "temperature": 1.0 / (self.kappa * self.beta),
            "refinement_policy": self.refinement_policy,
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
            "search_mode": self.search_mode,
            "beta": self.beta,
            "kappa": self.kappa,
            "temperature": 1.0 / (self.kappa * self.beta),
            "adaptive_c": self.adaptive_c,
            "refinement_policy": self.refinement_policy,
            "U": self.U,
            "entropy": self.entropy,
            "N_eff": self.N_eff,
            "K": self.K,
            "selected_depth": self.selected_depth,
            "selected_uci": list(self.selected_uci),
            "moves": [move.as_dict() for move in self.moves],
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SearchResult":
        return cls(
            board_fen=str(data["board_fen"]),
            player=str(data.get("player", "")),
            side=str(data.get("side", "")),
            cdepth=int(data.get("cdepth", 0)),
            search_mode=_validate_search_mode(str(data.get("search_mode", "accurate"))),
            beta=float(data.get("beta", 1.0)),
            kappa=float(data.get("kappa", KAPPA)),
            adaptive_c=float(data.get("adaptive_c", 0.3)),
            refinement_policy=_validate_refinement_policy(str(data.get("refinement_policy", "static_eval"))),
            U=float(data.get("U", data.get("expected_value", 0.0))),
            entropy=float(data.get("entropy", 0.0)),
            N_eff=float(data.get("N_eff", data.get("effective_moves", 0.0))),
            K=int(data.get("K", data.get("expanded_count", 0))),
            selected_depth=int(data.get("selected_depth", 0)),
            selected_uci=tuple(str(uci) for uci in data.get("selected_uci", [])),
            moves=tuple(MoveEvaluation.from_dict(item) for item in data.get("moves", [])),
            diagnostics=dict(data.get("diagnostics", {})),
        )


@dataclass(frozen=True)
class NodeSelection:
    entropy: float
    effective_moves: float
    selected_depth: int
    expanded_count: int
    selected_uci: Tuple[str, ...]
    branches: Tuple[AdaptiveBranchObservation, ...]


@dataclass
class SearchDiagnostics:
    requested_depth: int
    adaptive_c: float
    search_mode: SearchMode = "accurate"
    requested_cdepth: int = 0
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


def _terminal_observation(value: float, cdepth: int, search_mode: SearchMode) -> NodeSelection:
    branch = AdaptiveBranchObservation(
        move=chess.Move.null(),
        uci="__terminal__",
        probability=1.0,
        adaptive_branch_value=value,
        was_deepened=False,
        depth_used=0,
        search_mode=search_mode,
        cdepth=cdepth,
    )
    return NodeSelection(0.0, 1.0, 0, 0, (), (branch,))


def _branch_worker(args: Dict[str, object]) -> tuple[AdaptiveBranchObservation, Dict[str, float | int | str]]:
    evaluator = StaticEvaluator(args["eval_weights"])  # type: ignore[arg-type]
    search = AdaptiveExpectedValue(
        args["style"],  # type: ignore[arg-type]
        float(args["beta"]),
        evaluator,
        cdepth=int(args["cdepth"]),
        adaptive_c=float(args["adaptive_c"]),
        depth_thresholds=args["depth_thresholds"],  # type: ignore[arg-type]
        search_mode=_validate_search_mode(str(args["search_mode"])),
        search_workers=1,
        parallel_min_branches=int(args["parallel_min_branches"]),
        refinement_policy=_validate_refinement_policy(str(args.get("refinement_policy", "static_eval"))),
        kappa=float(args.get("kappa", KAPPA)),
    )
    search.begin_diagnostics()
    board = chess.Board(str(args["fen"]))
    record = args["record"]  # type: ignore[assignment]
    branch = search._evaluate_branch(  # noqa: SLF001 - process worker for this class.
        board,
        record,  # type: ignore[arg-type]
        set(args["selected"]),  # type: ignore[arg-type]
        int(args["selected_depth"]),
        int(args["cdepth"]),
        int(args["level"]),
    )
    return branch, search.finish_diagnostics()


class AdaptiveExpectedValue:
    """Compute entropy-adaptive expectations over full interaction cycles."""

    def __init__(
        self,
        style: Style,
        beta: float,
        evaluator: StaticEvaluator,
        cdepth: int = 1,
        adaptive_c: float = 0.3,
        depth_thresholds: AdaptiveDepthThresholds | None = None,
        search_mode: SearchMode = "accurate",
        search_workers: int | None = 1,
        parallel_min_branches: int = 8,
        candidate_thermo_mode: CandidateThermoMode = "refined",
        refinement_policy: RefinementPolicy = "static_eval",
        kappa: float = KAPPA,
        *,
        depth: int | None = None,
    ) -> None:
        if depth is not None:
            cdepth = depth
        if cdepth < 0:
            raise ValueError("cdepth must be at least 0")
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
        if candidate_thermo_mode not in {"selected", "refined", "all"}:
            raise ValueError("candidate_thermo_mode must be 'selected', 'refined', or 'all'")
        refinement_policy = _validate_refinement_policy(refinement_policy)
        self.style = style
        self.beta = beta
        self.kappa = kappa
        self.evaluator = evaluator
        self.cdepth = cdepth
        self.depth = cdepth
        self.adaptive_c = adaptive_c
        self.depth_thresholds = depth_thresholds or AdaptiveDepthThresholds()
        self.search_mode = _validate_search_mode(search_mode)
        self.search_workers = search_workers
        self.parallel_min_branches = parallel_min_branches
        self.candidate_thermo_mode = candidate_thermo_mode
        self.refinement_policy = refinement_policy
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
            cdepth,
            adaptive_c,
            self.search_mode,
            cdepth,
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
            self.cdepth,
            self.adaptive_c,
            self.search_mode,
            self.cdepth,
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
        remaining = self.cdepth if depth is None else depth
        return self._selection_cache.get((self.position_key(board), remaining))

    def branch_observations(
        self, board: chess.Board, depth: int | None = None
    ) -> Tuple[AdaptiveBranchObservation, ...]:
        remaining = self.cdepth if depth is None else depth
        self.expected_value(board, remaining)
        selection = self.node_selection(board, remaining)
        return selection.branches if selection is not None else ()

    def expected_value(self, board: chess.Board, depth: int | None = None) -> float:
        remaining = self.cdepth if depth is None else depth
        if remaining < 0:
            raise ValueError("cdepth must be at least 0")
        return self._expected_value(board, remaining, 0)

    def _future_subjective_value(self, board: chess.Board, remaining_cdepth: int, level: int) -> float:
        """Value at a same-player future state after one cycle was consumed.

        Public ``expected_value(..., 0)`` intentionally remains the static board
        value ``E(B)``. Candidate move scoring has different depth-zero
        semantics: the observer is to move again, so depth zero is the
        probability-weighted value of its immediate legal moves.
        """

        if remaining_cdepth < 0:
            raise ValueError("remaining_cdepth must be at least 0")
        if remaining_cdepth > 0:
            return self._expected_value(board, remaining_cdepth, level)

        self.diagnostics.maximum_depth_reached = max(
            self.diagnostics.maximum_depth_reached, level
        )
        position_key = self.position_key(board)
        cache_key = (position_key, remaining_cdepth)
        if cache_key in self._future_value_cache:
            self.diagnostics.cache_hits += 1
            return self._future_value_cache[cache_key]

        self.diagnostics.nodes_evaluated += 1
        if board.is_game_over(claim_draw=False):
            value = self.static_value(board)
            self._future_selection_cache[cache_key] = _terminal_observation(
                value, remaining_cdepth, self.search_mode
            )
            self._future_value_cache[cache_key] = value
            return value

        landscape = self.landscape(board)
        if not landscape.records:
            value = self.static_value(board)
            self._future_selection_cache[cache_key] = _terminal_observation(
                value, remaining_cdepth, self.search_mode
            )
            self._future_value_cache[cache_key] = value
            return value

        branches = tuple(
            AdaptiveBranchObservation(
                move=record.move,
                uci=record.uci,
                probability=record.probability,
                adaptive_branch_value=record.static_after,
                was_deepened=False,
                depth_used=0,
                search_mode=self.search_mode,
                cdepth=remaining_cdepth,
            )
            for record in landscape.records
        )
        self._future_selection_cache[cache_key] = NodeSelection(
            entropy=landscape.entropy,
            effective_moves=landscape.effective_moves,
            selected_depth=0,
            expanded_count=0,
            selected_uci=(),
            branches=branches,
        )
        self._future_value_cache[cache_key] = landscape.expected_value
        return landscape.expected_value

    def _future_branch_observations(
        self, board: chess.Board, remaining_cdepth: int, level: int
    ) -> Tuple[AdaptiveBranchObservation, ...]:
        if remaining_cdepth > 0:
            self._expected_value(board, remaining_cdepth, level)
            selection = self.node_selection(board, remaining_cdepth)
        else:
            self._future_subjective_value(board, remaining_cdepth, level)
            selection = self._future_selection_cache.get(
                (self.position_key(board), remaining_cdepth)
            )
        return selection.branches if selection is not None else ()

    def search_result(self, board: chess.Board, player: str = "", side: str | None = None) -> SearchResult:
        result_side = side or ("white" if board.turn == chess.WHITE else "black")
        key = (self.position_key(board), self.cdepth, result_side)
        cached = self._result_cache.get(key)
        if cached is not None:
            self.diagnostics.search_result_reuse_count += 1
            return cached
        self.expected_value(board, self.cdepth)
        selection = self.node_selection(board, self.cdepth)
        landscape = self.landscape(board)
        if selection is None:
            moves: Tuple[MoveEvaluation, ...] = ()
            selected_uci: Tuple[str, ...] = ()
            selected_depth = 0
            expanded_count = 0
        else:
            branch_by_uci = {branch.uci: branch for branch in selection.branches}
            current_u = self._future_subjective_value(board, 0, 0)
            current_branches = self._future_branch_observations(board, 0, 0)
            fast_thermos = {}
            for record in landscape.records:
                branch = branch_by_uci.get(record.uci)
                fast_thermos[record.uci] = self._candidate_thermodynamics(
                    board, record, branch, current_u, current_branches, detailed=False
                )

            if board.turn == chess.WHITE:
                chosen_uci = max(
                    landscape.records,
                    key=lambda record: fast_thermos[record.uci]["expected_delta_u_star"],
                ).uci
            else:
                chosen_uci = min(
                    landscape.records,
                    key=lambda record: fast_thermos[record.uci]["expected_delta_u_star"],
                ).uci

            if self.candidate_thermo_mode == "all":
                detailed_uci = {record.uci for record in landscape.records}
            elif self.candidate_thermo_mode == "refined":
                detailed_uci = set(selection.selected_uci)
            else:
                detailed_uci = {chosen_uci}

            moves_list = []
            for record in landscape.records:
                branch = branch_by_uci.get(record.uci)
                branch_value = branch.adaptive_branch_value if branch else record.static_after
                thermo = fast_thermos[record.uci]
                if record.uci in detailed_uci:
                    detailed_thermo = self._candidate_thermodynamics(
                        board, record, branch, current_u, current_branches, detailed=True
                    )
                    thermo = {**thermo, **detailed_thermo}
                    thermo["expected_next_U"] = fast_thermos[record.uci]["expected_next_U"]
                    thermo["expected_delta_u_star"] = fast_thermos[record.uci]["expected_delta_u_star"]
                    thermo["candidate_delta_u"] = fast_thermos[record.uci]["candidate_delta_u"]
                    thermo["candidate_delta_u_star"] = fast_thermos[record.uci]["candidate_delta_u_star"]
                    thermo["shallow_expected_delta_u_star"] = detailed_thermo["shallow_expected_delta_u_star"]
                moves_list.append(
                    MoveEvaluation(
                        move=record.move,
                        san=record.san,
                        uci=record.uci,
                        probability=record.probability,
                        potential=record.phi,
                        branch_value=branch_value,
                        candidate_delta_u=thermo["candidate_delta_u"],
                        selected_for_refinement=branch.was_deepened if branch else False,
                        static_value_after_move=record.static_after,
                        base_potential=record.base_potential,
                        phase_potential=record.phase_potential,
                        features=record.features,
                        expected_next_U=thermo["expected_next_U"],
                        expected_delta_u_star=thermo["expected_delta_u_star"],
                        candidate_delta_q=thermo["candidate_delta_q"],
                        candidate_delta_w=thermo["candidate_delta_w"],
                        candidate_delta_a=thermo["candidate_delta_a"],
                        candidate_delta_u_star=thermo["candidate_delta_u_star"],
                        shallow_expected_delta_u_star=thermo["shallow_expected_delta_u_star"],
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
            selected_depth = selection.selected_depth
            expanded_count = selection.expanded_count
        result = SearchResult(
            board_fen=board.fen(),
            player=player,
            side=result_side,
            cdepth=self.cdepth,
            search_mode=self.search_mode,
            beta=self.beta,
            adaptive_c=self.adaptive_c,
            refinement_policy=self.refinement_policy,
            kappa=self.kappa,
            U=self._future_subjective_value(board, 0, 0),
            entropy=landscape.entropy,
            N_eff=landscape.effective_moves,
            K=expanded_count,
            selected_depth=selected_depth,
            selected_uci=selected_uci,
            moves=moves,
            diagnostics=self.diagnostics.as_dict(),
        )
        self._result_cache[key] = result
        return result

    def _candidate_thermodynamics(
        self,
        board: chess.Board,
        record: MoveRecord,
        branch: AdaptiveBranchObservation | None,
        current_u: float,
        current_branches: Tuple[AdaptiveBranchObservation, ...],
        *,
        detailed: bool,
    ) -> Dict[str, object]:
        from .thermodynamics import decompose_transition

        branch_value = branch.adaptive_branch_value if branch else record.static_after
        after = board.copy(stack=False)
        after.push(record.move)
        if after.is_game_over(claim_draw=False):
            delta = record.static_after - current_u
            return {
                "expected_next_U": record.static_after,
                "expected_delta_u_star": delta,
                "candidate_delta_u": delta,
                "candidate_delta_u_star": delta,
                "shallow_expected_delta_u_star": delta if detailed else None,
                "candidate_delta_q": 0.0 if detailed else None,
                "candidate_delta_w": 0.0 if detailed else None,
                "candidate_delta_a": delta if detailed else None,
                "thermo_decomposition_error": 0.0 if detailed else None,
                "response_entropy": None,
                "response_N_eff": None,
                "response_K": None,
                "responses": (),
            }

        if (
            branch is None
            or not branch.response_branches
            or (not detailed and not branch.was_deepened)
        ):
            delta = branch_value - current_u
            return {
                "expected_next_U": branch_value,
                "expected_delta_u_star": delta,
                "candidate_delta_u": delta,
                "candidate_delta_u_star": delta,
                "shallow_expected_delta_u_star": None,
                "candidate_delta_q": None,
                "candidate_delta_w": None,
                "candidate_delta_a": None,
                "thermo_decomposition_error": None,
                "response_entropy": None,
                "response_N_eff": branch.response_neff if branch else None,
                "response_K": branch.response_k if branch else None,
                "responses": (),
            }

        response_records = branch.response_branches
        selected_uci = {
            response.uci for response in response_records if response.selected_for_refinement
        }

        response_evaluations: list[ResponseEvaluation] = []
        expected_next_u = 0.0
        candidate_delta_q = 0.0
        candidate_delta_w = 0.0
        candidate_delta_a = 0.0
        candidate_delta_u_star = 0.0
        for response in response_records:
            response_selected = response.uci in selected_uci
            next_u = response.response_value
            expected_next_u += response.probability * next_u

            if detailed:
                next_board = after.copy(stack=False)
                next_board.push(response.move)
                diagnostic_next_u = self._future_subjective_value(next_board, 0, level=1)
                next_branches = self._future_branch_observations(next_board, 0, level=1)
                decomposition = decompose_transition(
                    current_branches,
                    next_branches,
                    u_before=current_u,
                    u_after=diagnostic_next_u,
                )
                candidate_delta_u_star += response.probability * decomposition.delta_u
                candidate_delta_q += response.probability * decomposition.delta_q
                candidate_delta_w += response.probability * decomposition.delta_w
                candidate_delta_a += response.probability * decomposition.delta_a
                response_evaluations.append(
                    ResponseEvaluation(
                        move=response.move,
                        uci=response.uci,
                        san=response.san,
                        probability=response.probability,
                        value=next_u,
                        selected_for_refinement=response_selected,
                        static_value=response.static_value,
                        refinement_rank=response.refinement_rank,
                        depth_used=response.depth_used if response_selected else 0,
                        next_state_U=diagnostic_next_u,
                        delta_U=decomposition.delta_u,
                        delta_Q=decomposition.delta_q,
                        delta_W=decomposition.delta_w,
                        delta_A=decomposition.delta_a,
                        thermo_decomposition_error=decomposition.decomposition_error,
                        used_shallow_landscape=True,
                    )
                )

        expected_delta_u_star = expected_next_u - current_u
        if detailed:
            error = candidate_delta_u_star - (candidate_delta_q + candidate_delta_w + candidate_delta_a)
            delta_q = candidate_delta_q
            delta_w = candidate_delta_w
            delta_a = candidate_delta_a
            shallow_delta_u_star = candidate_delta_u_star
        else:
            error = None
            delta_q = None
            delta_w = None
            delta_a = None
            shallow_delta_u_star = None
        return {
            "expected_next_U": expected_next_u,
            "expected_delta_u_star": expected_delta_u_star,
            "candidate_delta_u": expected_delta_u_star,
            "candidate_delta_u_star": expected_delta_u_star,
            "shallow_expected_delta_u_star": shallow_delta_u_star,
            "candidate_delta_q": delta_q,
            "candidate_delta_w": delta_w,
            "candidate_delta_a": delta_a,
            "thermo_decomposition_error": error,
            "response_entropy": None,
            "response_N_eff": branch.response_neff,
            "response_K": branch.response_k,
            "responses": tuple(response_evaluations),
        }

    def _expected_value(self, board: chess.Board, cdepth: int, level: int) -> float:
        self.diagnostics.maximum_depth_reached = max(
            self.diagnostics.maximum_depth_reached, level
        )
        position_key = self.position_key(board)
        cache_key = (position_key, cdepth)
        if cache_key in self._value_cache:
            self.diagnostics.cache_hits += 1
            return self._value_cache[cache_key]

        self.diagnostics.nodes_evaluated += 1
        if cdepth == 0:
            value = self.static_value(board)
            self._value_cache[cache_key] = value
            return value
        if board.is_game_over(claim_draw=False):
            value = self.static_value(board)
            self._selection_cache[cache_key] = _terminal_observation(value, cdepth, self.search_mode)
            self._value_cache[cache_key] = value
            return value

        landscape = self.landscape(board)
        if not landscape.records:
            value = self.static_value(board)
            self._selection_cache[cache_key] = _terminal_observation(value, cdepth, self.search_mode)
            self._value_cache[cache_key] = value
            return value

        expanded_count = adaptive_breadth(
            landscape.effective_moves, self.adaptive_c, len(landscape.records)
        )
        selected_depth = local_search_depth(
            cdepth, landscape.effective_moves, self.depth_thresholds
        )
        ranked = ranked_for_refinement(landscape.records, board.turn, self.refinement_policy)
        selected_uci = tuple(record.uci for record in ranked[:expanded_count])
        selected = set(selected_uci)
        self.diagnostics.expanded_total += expanded_count
        self.diagnostics.expanded_nodes += 1

        branches = self._evaluate_branches(board, landscape.records, selected, selected_depth, cdepth, level)
        value = sum(branch.probability * branch.adaptive_branch_value for branch in branches)

        self._selection_cache[cache_key] = NodeSelection(
            entropy=landscape.entropy,
            effective_moves=landscape.effective_moves,
            selected_depth=selected_depth,
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
        selected_depth: int,
        cdepth: int,
        level: int,
    ) -> Tuple[AdaptiveBranchObservation, ...]:
        records = tuple(records)
        should_parallel = (
            level == 0
            and self.search_workers > 1
            and len(records) >= self.parallel_min_branches
            and cdepth > 0
        )
        if not should_parallel:
            return tuple(
                self._evaluate_branch(board, record, selected, selected_depth, cdepth, level)
                for record in records
            )
        args = [
            {
                "fen": board.fen(),
                "record": record,
                "selected": tuple(selected),
                "selected_depth": selected_depth,
                "cdepth": cdepth,
                "level": level,
                "style": self.style,
                "beta": self.beta,
                "adaptive_c": self.adaptive_c,
                "depth_thresholds": self.depth_thresholds,
                "search_mode": self.search_mode,
                "eval_weights": self.evaluator.weights,
                "parallel_min_branches": self.parallel_min_branches,
                "refinement_policy": self.refinement_policy,
                "kappa": self.kappa,
            }
            for record in records
        ]
        self.diagnostics.parallel_branches += len(records)
        with ProcessPoolExecutor(max_workers=self.search_workers) as executor:
            results = list(executor.map(_branch_worker, args))
        branches = []
        for branch, child_diag in results:
            branches.append(branch)
            child = SearchDiagnostics(
                self.cdepth,
                self.adaptive_c,
                self.search_mode,
                self.cdepth,
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
        selected_depth: int,
        cdepth: int,
        level: int,
    ) -> AdaptiveBranchObservation:
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
            reply_landscape = self.response_landscape(after)
            response_neff = reply_landscape.effective_moves
            response_k = adaptive_breadth(reply_landscape.effective_moves, self.adaptive_c, len(reply_landscape.records))
            reply_ranked = ranked_for_refinement(reply_landscape.records, after.turn, self.refinement_policy)
            selected_reply_rank = {reply.uci: rank for rank, reply in enumerate(reply_ranked[:response_k], start=1)}
            selected_replies = set(selected_reply_rank)
            self.diagnostics.response_expanded_total += response_k
            self.diagnostics.response_expanded_nodes += 1
            branch_value = 0.0
            for reply in reply_landscape.records:
                reply_selected = reply.uci in selected_replies
                reply_value = reply.static_after
                used_shallow_landscape = False
                response_depth_used = 0
                if own_selected and reply_selected:
                    child = after.copy(stack=False)
                    child.push(reply.move)
                    remaining_cycles = max(0, selected_depth - 1)
                    if remaining_cycles > 0:
                        self.diagnostics.recursive_nodes += 1
                        reply_value = self._expected_value(child, remaining_cycles, level + 1)
                        response_depth_used = remaining_cycles
                    else:
                        reply_value = self._future_subjective_value(child, 0, level + 1)
                        used_shallow_landscape = True
                branch_value += reply.probability * reply_value
                response_branches.append(
                    ResponseObservation(
                        move=reply.move,
                        uci=reply.uci,
                        san=after.san(reply.move),
                        probability=reply.probability,
                        response_value=reply_value,
                        selected_for_refinement=reply_selected,
                        depth_used=response_depth_used,
                        static_value=reply.static_after,
                        refinement_rank=selected_reply_rank.get(reply.uci),
                        used_shallow_landscape=used_shallow_landscape,
                    )
                )
        return AdaptiveBranchObservation(
            move=record.move,
            uci=record.uci,
            probability=record.probability,
            adaptive_branch_value=branch_value,
            was_deepened=own_selected,
            depth_used=selected_depth if own_selected else 0,
            response_neff=response_neff,
            response_k=response_k,
            response_branches=tuple(response_branches),
            search_mode=self.search_mode,
            cdepth=cdepth,
        )
