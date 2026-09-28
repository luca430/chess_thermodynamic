"""Artificial players using style-dependent adaptive expectations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import chess

from .evaluation import StaticEvaluator
from .measure import MoveLandscape, Style
from .search import AdaptiveDepthThresholds, AdaptiveExpectedValue, PositionKey


@dataclass(frozen=True)
class CandidateScore:
    move: chess.Move
    san: str
    uci: str
    probability: float
    static_after: float
    value: float
    delta_u: float
    selected_for_deeper_analysis: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "san": self.san,
            "uci": self.uci,
            "probability": self.probability,
            "static_after": self.static_after,
            "selection_value": self.value,
            "delta_u": self.delta_u,
            "selected_for_deeper_analysis": self.selected_for_deeper_analysis,
        }


@dataclass(frozen=True)
class PositionAnalysis:
    current_value: float
    landscape: MoveLandscape
    candidates: tuple[CandidateScore, ...]
    entropy: float
    effective_moves: float
    selected_depth: int
    expanded_count: int
    selected_uci: tuple[str, ...]
    diagnostics: Dict[str, float | int]


@dataclass(frozen=True)
class Choice:
    move: Optional[chess.Move]
    san: str
    uci: str
    value: float
    current_value: float
    delta_u: float
    current_landscape: MoveLandscape
    reply_landscape: MoveLandscape | None
    analysis: PositionAnalysis


@dataclass
class ThermoPlayer:
    name: str
    color: chess.Color
    style: Style
    beta: float = 4.0
    depth: int = 4
    adaptive_c: float = 0.3
    depth_thresholds: AdaptiveDepthThresholds = field(
        default_factory=AdaptiveDepthThresholds
    )
    _search: AdaptiveExpectedValue | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _evaluator_id: int | None = field(default=None, init=False, repr=False, compare=False)
    _analysis_cache: Dict[PositionKey, PositionAnalysis] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError("depth must be at least 0")
        if self.adaptive_c <= 0.0:
            raise ValueError("adaptive_c must be greater than 0")

    def search(self, evaluator: StaticEvaluator) -> AdaptiveExpectedValue:
        if self._search is None or self._evaluator_id != id(evaluator):
            self._search = AdaptiveExpectedValue(
                self.style,
                self.beta,
                evaluator,
                depth=self.depth,
                adaptive_c=self.adaptive_c,
                depth_thresholds=self.depth_thresholds,
            )
            self._evaluator_id = id(evaluator)
            self._analysis_cache.clear()
        return self._search

    def landscape(self, board: chess.Board, evaluator: StaticEvaluator) -> MoveLandscape:
        return self.search(evaluator).landscape(board)

    def analyze(self, board: chess.Board, evaluator: StaticEvaluator) -> PositionAnalysis:
        search = self.search(evaluator)
        key = search.position_key(board)
        cached = self._analysis_cache.get(key)
        if cached is not None:
            return cached

        search.begin_diagnostics()
        landscape = search.landscape(board)
        current_value = search.expected_value(board)
        selection = search.node_selection(board)
        selected_uci = selection.selected_uci if selection else ()
        selected = set(selected_uci)

        candidates = []
        for record in landscape.records:
            after = board.copy(stack=False)
            after.push(record.move)
            value = search.expected_value(after)
            candidates.append(
                CandidateScore(
                    move=record.move,
                    san=record.san,
                    uci=record.uci,
                    probability=record.probability,
                    static_after=record.static_after,
                    value=value,
                    delta_u=value - current_value,
                    selected_for_deeper_analysis=record.uci in selected,
                )
            )

        analysis = PositionAnalysis(
            current_value=current_value,
            landscape=landscape,
            candidates=tuple(candidates),
            entropy=landscape.entropy,
            effective_moves=landscape.effective_moves,
            selected_depth=selection.selected_depth if selection else 0,
            expanded_count=selection.expanded_count if selection else 0,
            selected_uci=selected_uci,
            diagnostics=search.finish_diagnostics(),
        )
        self._analysis_cache[key] = analysis
        return analysis

    def choose(self, board: chess.Board, evaluator: StaticEvaluator) -> Choice:
        analysis = self.analyze(board, evaluator)
        if not analysis.candidates:
            return Choice(
                None,
                "",
                "",
                analysis.current_value,
                analysis.current_value,
                0.0,
                analysis.landscape,
                None,
                analysis,
            )

        chooser = max if self.color == chess.WHITE else min
        best = chooser(analysis.candidates, key=lambda candidate: candidate.value)
        after = board.copy(stack=False)
        after.push(best.move)
        return Choice(
            move=best.move,
            san=best.san,
            uci=best.uci,
            value=best.value,
            current_value=analysis.current_value,
            delta_u=best.delta_u,
            current_landscape=analysis.landscape,
            reply_landscape=self.landscape(after, evaluator),
            analysis=analysis,
        )
