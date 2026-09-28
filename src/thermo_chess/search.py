"""Entropy-adaptive recursive expected values over complete move-response cycles."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, Literal, Tuple

import chess

from .evaluation import StaticEvaluator
from .measure import MoveLandscape, Style, move_distribution


PositionKey = Tuple[str, bool, bool, bool]
SearchMode = Literal["accurate", "cheap"]


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


def _validate_search_mode(search_mode: str) -> SearchMode:
    if search_mode not in {"accurate", "cheap"}:
        raise ValueError("search_mode must be 'accurate' or 'cheap'")
    return search_mode  # type: ignore[return-value]


@dataclass(frozen=True)
class ResponseObservation:
    move: chess.Move
    uci: str
    probability: float
    response_value: float
    selected_for_refinement: bool
    depth_used: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "uci": self.uci,
            "probability": self.probability,
            "response_value": self.response_value,
            "selected_for_refinement": self.selected_for_refinement,
            "depth_used": self.depth_used,
        }


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
            "depth_used": self.depth_used,
            "cdepth": self.cdepth,
            "search_mode": self.search_mode,
            "response_neff": self.response_neff,
            "response_k": self.response_k,
            "response_branches": [branch.as_dict() for branch in self.response_branches],
        }


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
    nodes_evaluated: int = 0
    static_evaluations: int = 0
    recursive_nodes: int = 0
    maximum_depth_reached: int = 0
    cache_hits: int = 0
    expanded_total: int = 0
    expanded_nodes: int = 0
    response_expanded_total: int = 0
    response_expanded_nodes: int = 0
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

    def evaluate(self, board: chess.Board) -> float:
        return self.search.static_value(board)


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
        *,
        depth: int | None = None,
    ) -> None:
        if depth is not None:
            cdepth = depth
        if cdepth < 0:
            raise ValueError("cdepth must be at least 0")
        if adaptive_c <= 0.0:
            raise ValueError("adaptive_c must be greater than 0")
        self.style = style
        self.beta = beta
        self.evaluator = evaluator
        self.cdepth = cdepth
        self.depth = cdepth
        self.adaptive_c = adaptive_c
        self.depth_thresholds = depth_thresholds or AdaptiveDepthThresholds()
        self.search_mode = _validate_search_mode(search_mode)
        self._static_cache: Dict[PositionKey, float] = {}
        self._landscape_cache: Dict[PositionKey, MoveLandscape] = {}
        self._value_cache: Dict[Tuple[PositionKey, int], float] = {}
        self._selection_cache: Dict[Tuple[PositionKey, int], NodeSelection] = {}
        self._cached_evaluator = _CachedEvaluator(self)
        self.diagnostics = SearchDiagnostics(cdepth, adaptive_c, self.search_mode, cdepth)
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
            self.cdepth, self.adaptive_c, self.search_mode, self.cdepth
        )
        self._started_at = time.perf_counter()

    def finish_diagnostics(self) -> Dict[str, float | int | str]:
        self.diagnostics.elapsed_time = time.perf_counter() - self._started_at
        return self.diagnostics.as_dict()

    def static_value(self, board: chess.Board) -> float:
        key = self.position_key(board)
        if key in self._static_cache:
            self.diagnostics.cache_hits += 1
            return self._static_cache[key]
        value = self.evaluator.evaluate(board)
        self._static_cache[key] = value
        self.diagnostics.static_evaluations += 1
        return value

    def landscape(self, board: chess.Board) -> MoveLandscape:
        key = self.position_key(board)
        if key in self._landscape_cache:
            self.diagnostics.cache_hits += 1
            return self._landscape_cache[key]
        landscape = move_distribution(
            board, self.style, self.beta, self._cached_evaluator
        )
        self._landscape_cache[key] = landscape
        return landscape

    def node_selection(self, board: chess.Board, depth: int | None = None) -> NodeSelection | None:
        remaining = self.cdepth if depth is None else depth
        return self._selection_cache.get((self.position_key(board), remaining))

    def branch_observations(
        self, board: chess.Board, depth: int | None = None
    ) -> Tuple[AdaptiveBranchObservation, ...]:
        """Return the exact branch terms used by the cached adaptive expectation."""

        remaining = self.cdepth if depth is None else depth
        self.expected_value(board, remaining)
        selection = self.node_selection(board, remaining)
        return selection.branches if selection is not None else ()

    def expected_value(self, board: chess.Board, depth: int | None = None) -> float:
        remaining = self.cdepth if depth is None else depth
        if remaining < 0:
            raise ValueError("cdepth must be at least 0")
        return self._expected_value(board, remaining, 0)

    def _terminal_observation(self, board: chess.Board, value: float, cdepth: int) -> NodeSelection:
        branch = AdaptiveBranchObservation(
            move=chess.Move.null(),
            uci="__terminal__",
            probability=1.0,
            adaptive_branch_value=value,
            was_deepened=False,
            depth_used=0,
            search_mode=self.search_mode,
            cdepth=cdepth,
        )
        return NodeSelection(0.0, 1.0, 0, 0, (), (branch,))

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
        if board.is_game_over(claim_draw=True):
            value = self.static_value(board)
            self._selection_cache[cache_key] = self._terminal_observation(board, value, cdepth)
            self._value_cache[cache_key] = value
            return value

        landscape = self.landscape(board)
        if not landscape.records:
            value = self.static_value(board)
            self._selection_cache[cache_key] = self._terminal_observation(board, value, cdepth)
            self._value_cache[cache_key] = value
            return value

        expanded_count = adaptive_breadth(
            landscape.effective_moves, self.adaptive_c, len(landscape.records)
        )
        selected_depth = local_search_depth(
            cdepth, landscape.effective_moves, self.depth_thresholds
        )
        ranked = sorted(
            landscape.records,
            key=lambda record: record.probability,
            reverse=True,
        )
        selected_uci = tuple(record.uci for record in ranked[:expanded_count])
        selected = set(selected_uci)
        self.diagnostics.expanded_total += expanded_count
        self.diagnostics.expanded_nodes += 1

        value = 0.0
        branches = []
        for record in landscape.records:
            own_selected = record.uci in selected
            branch_value = record.static_after
            response_neff = None
            response_k = None
            response_branches: list[ResponseObservation] = []

            after = board.copy(stack=False)
            after.push(record.move)
            if after.is_game_over(claim_draw=True):
                branch_value = self.static_value(after)
            elif self.search_mode == "cheap" and not own_selected:
                branch_value = record.static_after
            else:
                reply_landscape = self.landscape(after)
                response_neff = reply_landscape.effective_moves
                response_k = adaptive_breadth(
                    reply_landscape.effective_moves,
                    self.adaptive_c,
                    len(reply_landscape.records),
                )
                reply_ranked = sorted(
                    reply_landscape.records,
                    key=lambda reply: reply.probability,
                    reverse=True,
                )
                selected_replies = {reply.uci for reply in reply_ranked[:response_k]}
                self.diagnostics.response_expanded_total += response_k
                self.diagnostics.response_expanded_nodes += 1
                branch_value = 0.0
                selected_mass = 0.0
                for reply in reply_landscape.records:
                    reply_selected = reply.uci in selected_replies
                    reply_value = reply.static_after
                    if own_selected and reply_selected and selected_depth > 1:
                        child = after.copy(stack=False)
                        child.push(reply.move)
                        self.diagnostics.recursive_nodes += 1
                        reply_value = self._expected_value(
                            child, selected_depth - 1, level + 1
                        )
                    elif self.search_mode == "cheap" and not reply_selected:
                        continue
                    elif reply_selected or self.search_mode == "accurate":
                        reply_value = reply.static_after
                    if self.search_mode == "cheap" and reply_selected:
                        selected_mass += reply.probability
                    branch_value += reply.probability * reply_value
                    response_branches.append(
                        ResponseObservation(
                            move=reply.move,
                            uci=reply.uci,
                            probability=reply.probability,
                            response_value=reply_value,
                            selected_for_refinement=reply_selected,
                            depth_used=selected_depth - 1
                            if own_selected and reply_selected and selected_depth > 1
                            else 0,
                        )
                    )
                if self.search_mode == "cheap":
                    branch_value += max(0.0, 1.0 - selected_mass) * record.static_after

            branches.append(
                AdaptiveBranchObservation(
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
            )
            value += record.probability * branch_value

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
