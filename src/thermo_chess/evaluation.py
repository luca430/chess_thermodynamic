"""Universal static board evaluation E(B)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import chess

from .features import BoardFeatureContext, white_minus_black_features


@dataclass(frozen=True)
class EvaluationWeights:
    material: float = 1.0
    mobility: float = 0.25
    king_safety: float = 0.6
    center: float = 0.35
    checkmate: float = 10_000.0
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        weights = {
            "material": self.material,
            "mobility": self.mobility,
            "king_safety": self.king_safety,
            "center": self.center,
        }
        weights.update(self.extra)
        return weights


class StaticEvaluator:
    """Inexpensive non-recursive evaluator shared by all players."""

    def __init__(self, weights: EvaluationWeights | None = None):
        self.weights = weights or EvaluationWeights()

    def evaluate(self, board: chess.Board, context: BoardFeatureContext | None = None) -> float:
        if board.is_checkmate():
            # Side to move is checkmated.
            return -self.weights.checkmate if board.turn == chess.WHITE else self.weights.checkmate
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()
        ):
            return 0.0

        features = white_minus_black_features(board, context)
        weights = self.weights.as_dict()
        return sum(weights.get(name, 0.0) * value for name, value in features.items())
