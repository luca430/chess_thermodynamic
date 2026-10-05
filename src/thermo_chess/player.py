"""Artificial players using style-dependent adaptive expectations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import chess

from .evaluation import StaticEvaluator
from .measure import KAPPA, MoveLandscape, Style
from .search import AdaptiveExpectedValue, MoveEvaluation, PositionKey, SearchMode, SearchResult, validate_cdepth


@dataclass(frozen=True)
class CandidateScore:
    move: chess.Move
    san: str
    uci: str
    probability: float
    static_after: float
    value: float
    g_tilde: float | None
    selected_for_refinement: bool
    response_neff: float | None = None
    response_k: int | None = None
    search_mode: SearchMode = "accurate"
    cdepth: int = 0

    @classmethod
    def from_move_evaluation(cls, move: MoveEvaluation, result: SearchResult) -> "CandidateScore":
        return cls(
            move=move.move,
            san=move.san,
            uci=move.uci,
            probability=move.probability,
            static_after=move.static_value_after_move,
            value=move.terminal_u if move.terminal_u is not None else move.branch_value,
            g_tilde=move.g_tilde,
            selected_for_refinement=move.selected_for_refinement,
            response_neff=move.response_N_eff,
            response_k=move.response_K,
            search_mode=result.search_mode,
            cdepth=result.cdepth,
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "san": self.san,
            "uci": self.uci,
            "probability": self.probability,
            "static_after": self.static_after,
            "selection_value": self.value,
            "branch_value": self.value,
            "terminal_u": self.value,
            "g_tilde": self.g_tilde,
            "selected_for_refinement": self.selected_for_refinement,
            "response_neff": self.response_neff,
            "response_k": self.response_k,
            "search_mode": self.search_mode,
            "cdepth": self.cdepth,
        }


@dataclass(frozen=True)
class PositionAnalysis:
    current_value: float
    landscape: MoveLandscape
    candidates: tuple[CandidateScore, ...]
    entropy: float
    effective_moves: float
    expanded_count: int
    selected_uci: tuple[str, ...]
    diagnostics: Dict[str, float | int | str]
    search_result: SearchResult


@dataclass(frozen=True)
class Choice:
    move: Optional[chess.Move]
    san: str
    uci: str
    value: float
    current_value: float
    delta_u: float
    current_landscape: MoveLandscape | None
    reply_landscape: MoveLandscape | None
    analysis: PositionAnalysis
    search_result: SearchResult


@dataclass
class ThermoPlayer:
    name: str
    color: chess.Color
    style: Style
    beta: float = 4.0
    cdepth: int = 2
    adaptive_c: float = 0.3
    search_mode: SearchMode = "accurate"
    search_workers: int | None = 1
    parallel_min_branches: int = 8
    kappa: float = KAPPA
    _search: AdaptiveExpectedValue | None = field(default=None, init=False, repr=False, compare=False)
    _evaluator_id: int | None = field(default=None, init=False, repr=False, compare=False)
    _analysis_cache: Dict[PositionKey, PositionAnalysis] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.cdepth = validate_cdepth(self.cdepth)
        if self.kappa <= 0.0:
            raise ValueError("kappa must be greater than 0")
        if self.adaptive_c <= 0.0:
            raise ValueError("adaptive_c must be greater than 0")
        if self.search_mode != "accurate":
            raise ValueError("search_mode must be 'accurate'")
        if self.search_workers is not None and self.search_workers < 1:
            raise ValueError("search_workers must be at least 1")
        if self.parallel_min_branches < 1:
            raise ValueError("parallel_min_branches must be at least 1")

    def search(self, evaluator: StaticEvaluator) -> AdaptiveExpectedValue:
        if self._search is None or self._evaluator_id != id(evaluator):
            self._search = AdaptiveExpectedValue(
                self.style,
                self.beta,
                evaluator,
                kappa=self.kappa,
                cdepth=self.cdepth,
                adaptive_c=self.adaptive_c,
                search_mode=self.search_mode,
                search_workers=self.search_workers,
                parallel_min_branches=self.parallel_min_branches,
            )
            self._evaluator_id = id(evaluator)
            self._analysis_cache.clear()
        return self._search

    def landscape(self, board: chess.Board, evaluator: StaticEvaluator) -> MoveLandscape:
        return self.search(evaluator).landscape(board)

    def evaluate_landscape(self, board: chess.Board, evaluator: StaticEvaluator) -> SearchResult:
        return self.search(evaluator).search_result(
            board,
            player=self.name,
            side="white" if self.color == chess.WHITE else "black",
        )

    def analyze(self, board: chess.Board, evaluator: StaticEvaluator) -> PositionAnalysis:
        search = self.search(evaluator)
        key = search.position_key(board)
        cached = self._analysis_cache.get(key)
        if cached is not None:
            return cached

        search.begin_diagnostics()
        result = self.evaluate_landscape(board, evaluator)
        diagnostics = search.finish_diagnostics()
        result = SearchResult(
            board_fen=result.board_fen,
            player=result.player,
            side=result.side,
            cdepth=result.cdepth,
            requested_recursive_plies=result.requested_recursive_plies,
            search_mode=result.search_mode,
            beta=result.beta,
            adaptive_c=result.adaptive_c,
            kappa=result.kappa,
            U=result.U,
            entropy=result.entropy,
            N_eff=result.N_eff,
            K=result.K,
            selected_uci=result.selected_uci,
            moves=result.moves,
            diagnostics=diagnostics,
        )
        # Keep the cache authoritative for later consumers in this turn.
        search._result_cache[(key, search.requested_recursive_plies, result.side)] = result  # noqa: SLF001
        candidates = tuple(CandidateScore.from_move_evaluation(move, result) for move in result.moves)
        analysis = PositionAnalysis(
            current_value=result.U,
            landscape=search.landscape(board),
            candidates=candidates,
            entropy=result.entropy,
            effective_moves=result.N_eff,
            expanded_count=result.K,
            selected_uci=result.selected_uci,
            diagnostics=diagnostics,
            search_result=result,
        )
        self._analysis_cache[key] = analysis
        return analysis

    def choose_from_result(self, result: SearchResult) -> Choice:
        candidates = tuple(CandidateScore.from_move_evaluation(move, result) for move in result.moves)
        analysis = PositionAnalysis(
            current_value=result.U,
            landscape=MoveLandscape(
                fen=result.board_fen,
                turn=result.side,
                beta=result.beta,
                records=[],
                entropy=result.entropy,
                effective_moves=result.N_eff,
                expected_value=result.U,
            ),
            candidates=candidates,
            entropy=result.entropy,
            effective_moves=result.N_eff,
            expanded_count=result.K,
            selected_uci=result.selected_uci,
            diagnostics=result.diagnostics,
            search_result=result,
        )
        if not result.moves:
            return Choice(None, "", "", result.U, result.U, 0.0, None, None, analysis, result)
        selectable = tuple(
            move
            for move in result.moves
            if move.g_tilde is not None
        )
        if not selectable:
            return Choice(None, "", "", result.U, result.U, 0.0, None, None, analysis, result)
        chooser = max if self.color == chess.WHITE else min
        best = chooser(selectable, key=lambda move: move.g_tilde)
        return Choice(
            move=best.move,
            san=best.san,
            uci=best.uci,
            value=best.terminal_u if best.terminal_u is not None else best.branch_value,
            current_value=result.U,
            delta_u=best.g_tilde,
            current_landscape=None,
            reply_landscape=None,
            analysis=analysis,
            search_result=result,
        )

    def choose(self, board: chess.Board, evaluator: StaticEvaluator) -> Choice:
        analysis = self.analyze(board, evaluator)
        return self.choose_from_result(analysis.search_result)
