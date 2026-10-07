"""Universal static board evaluation E(B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import chess

from .features import BoardFeatureContext, white_minus_black_features

STATIC_EVALUATION_COMPONENTS = (
    "material",
    "pawn_structure",
    "mobility",
    "center",
    "king_safety",
)


@dataclass(frozen=True)
class EvaluationWeights:
    material: float = 1.0
    pawn_structure: float = 1.0
    mobility: float = 1.0
    center: float = 1.0
    king_safety: float = 1.0
    checkmate: float = 10_000.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "material": self.material,
            "pawn_structure": self.pawn_structure,
            "mobility": self.mobility,
            "center": self.center,
            "king_safety": self.king_safety,
        }


class StaticEvaluator:
    """Inexpensive non-recursive evaluator shared by all players."""

    def __init__(self, weights: EvaluationWeights | None = None):
        self.weights = weights or EvaluationWeights()

    def terminal_reason(self, board: chess.Board) -> str | None:
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
        return None

    def components(
        self, board: chess.Board, context: BoardFeatureContext | None = None
    ) -> Dict[str, float] | None:
        """White-positive pawn-like components of the ordinary static evaluator."""
        if self.terminal_reason(board) is not None:
            return None
        features = white_minus_black_features(board, context)
        return {name: float(features[name]) for name in STATIC_EVALUATION_COMPONENTS}

    def value_from_components(self, components: Dict[str, float] | None) -> float:
        if components is None:
            return 0.0
        weights = self.weights.as_dict()
        return sum(weights.get(name, 0.0) * value for name, value in components.items())

    def evaluate_with_components(
        self, board: chess.Board, context: BoardFeatureContext | None = None
    ) -> tuple[float, Dict[str, float] | None]:
        if board.is_checkmate():
            # Side to move is checkmated.
            value = -self.weights.checkmate if board.turn == chess.WHITE else self.weights.checkmate
            return value, None
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()
        ):
            return 0.0, None
        features = self.components(board, context)
        return self.value_from_components(features), features

    def evaluate(self, board: chess.Board, context: BoardFeatureContext | None = None) -> float:
        value, _components = self.evaluate_with_components(board, context=context)
        return value


def static_evaluation_components(
    board: chess.Board, context: BoardFeatureContext | None = None
) -> Dict[str, float] | None:
    return StaticEvaluator().components(board, context)
