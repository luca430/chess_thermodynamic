"""Player-dependent move potentials and probability measures."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List

import chess

from .evaluation import StaticEvaluator
from .features import BoardFeatureContext, board_context, move_features
from .metrics import effective_number, entropy


@dataclass(frozen=True)
class Style:
    material: float = 1.0
    preservation: float = 1.0
    king_restriction: float = 0.4
    king_pressure: float = 0.2
    check: float = 0.2
    mate: float = 50.0
    activity: float = 0.5
    king_safety: float = 0.5
    center: float = 0.3
    promotion: float = 1.0
    castle_preserve: float = 0.0
    castle_deny: float = 0.0
    castle: float = 0.0
    development: float = 0.0
    phase_castle: float = 0.0
    phase_attack: float = 0.0
    solidness: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.solidness <= 1.0:
            raise ValueError("solidness must be between 0 and 1")

    def as_dict(self) -> Dict[str, float]:
        weights = {
            "material": self.material,
            "preservation": self.preservation,
            "king_restriction": self.king_restriction,
            "king_pressure": self.king_pressure,
            "check": self.check,
            "mate": self.mate,
            "activity": self.activity,
            "king_safety": self.king_safety,
            "center": self.center,
            "promotion": self.promotion,
            "castle_preserve": self.castle_preserve,
            "castle_deny": self.castle_deny,
            "castle": self.castle,
            "development": self.development,
            "phase_castle": self.phase_castle,
            "phase_attack": self.phase_attack,
            "solidness": self.solidness,
        }
        weights.update(self.extra)
        return weights


@dataclass(frozen=True)
class MoveRecord:
    move: chess.Move
    san: str
    uci: str
    phi: float
    probability: float
    static_after: float
    features: Dict[str, float]
    base_potential: float
    phase_potential: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "san": self.san,
            "uci": self.uci,
            "phi": self.phi,
            "probability": self.probability,
            "static_after": self.static_after,
            "features": self.features,
            "base_potential": self.base_potential,
            "phase_potential": self.phase_potential,
            "total_potential": self.phi,
        }


@dataclass(frozen=True)
class MoveLandscape:
    fen: str
    turn: str
    beta: float
    records: List[MoveRecord]
    entropy: float
    effective_moves: float
    expected_value: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "fen": self.fen,
            "turn": self.turn,
            "beta": self.beta,
            "entropy": self.entropy,
            "effective_moves": self.effective_moves,
            "expected_value": self.expected_value,
            "moves": [record.as_dict() for record in self.records],
        }


def castle_preserve_diagnostics(style: Style, features: Dict[str, float]) -> Dict[str, float]:
    raw = features.get("castle_preserve", 0.0)
    phase = min(1.0, max(0.0, features.get("game_phase", 0.0)))
    rights_weight = 1.0 - phase
    castle_phase_weight = features.get("castle_phase_weight", 0.0)
    effective_weight = (
        style.castle_preserve
        * rights_weight
        * (1.0 + style.solidness * castle_phase_weight)
    )
    return {
        "castle_preserve_raw": raw,
        "castle_preserve_phase_weight": rights_weight,
        "castle_preserve_effective_weight": effective_weight,
        "castle_preserve_contribution": effective_weight * raw,
    }


def potential_components(style: Style, features: Dict[str, float]) -> tuple[float, float, float]:
    weights = style.as_dict()
    special_parameters = {
        "development",
        "phase_castle",
        "phase_attack",
        "solidness",
        "castle_preserve",
        "castle_preserve_raw",
        "castle_preserve_phase_weight",
        "castle_preserve_effective_weight",
        "castle_preserve_contribution",
    }
    castle_terms = castle_preserve_diagnostics(style, features)
    base = sum(
        weights.get(name, 0.0) * value
        for name, value in features.items()
        if name not in special_parameters
    ) + castle_terms["castle_preserve_contribution"]
    phase = (
        style.development
        * features.get("development_phase_weight", 0.0)
        * features.get("development_feature", 0.0)
        + style.phase_castle
        * features.get("castle_phase_weight", 0.0)
        * features.get("phase_castle_feature", 0.0)
        + style.phase_attack
        * features.get("attack_phase_weight", 0.0)
        * features.get("phase_attack_feature", 0.0)
    )
    return base, phase, base + style.solidness * phase


def potential(style: Style, features: Dict[str, float]) -> float:
    return potential_components(style, features)[2]


def stable_softmax(values: Iterable[float], beta: float) -> List[float]:
    scaled = [beta * value for value in values]
    if not scaled:
        return []
    max_value = max(scaled)
    exp_values = [math.exp(value - max_value) for value in scaled]
    normalizer = sum(exp_values)
    if normalizer == 0.0:
        return [1.0 / len(exp_values)] * len(exp_values)
    return [value / normalizer for value in exp_values]


def move_distribution(
    board: chess.Board,
    style: Style,
    beta: float,
    evaluator: StaticEvaluator,
    before_context: BoardFeatureContext | None = None,
    context_provider: Callable[[chess.Board], BoardFeatureContext] | None = None,
    value_objective_weight: float = 0.0,
) -> MoveLandscape:
    context_for = context_provider or board_context
    before = before_context or context_for(board)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        value = evaluator.evaluate(board, context=before)
        return MoveLandscape(
            fen=board.fen(),
            turn="white" if board.turn == chess.WHITE else "black",
            beta=beta,
            records=[],
            entropy=0.0,
            effective_moves=0.0,
            expected_value=value,
        )

    after_boards = []
    after_contexts = []
    features_by_move = []
    for move in legal_moves:
        after = board.copy(stack=False)
        after.push(move)
        after_values = context_for(after)
        after_boards.append(after)
        after_contexts.append(after_values)
        features_by_move.append(
            move_features(
                board,
                move,
                before_context=before,
                after_context=after_values,
                after_board=after,
            )
        )
    components = [potential_components(style, features) for features in features_by_move]
    static_values = [
        evaluator.evaluate(after, context=after_values)
        for after, after_values in zip(after_boards, after_contexts)
    ]
    side_sign = 1.0 if board.turn == chess.WHITE else -1.0
    phis = [
        total + value_objective_weight * side_sign * static_after
        for (_, _, total), static_after in zip(components, static_values)
    ]
    probabilities = stable_softmax(phis, beta)

    records: List[MoveRecord] = []
    expected = 0.0
    for move, after, after_values, features, (base_phi, phase_phi, _style_phi), phi, probability, static_after in zip(
        legal_moves, after_boards, after_contexts, features_by_move, components, phis, probabilities, static_values
    ):
        features = {**features, **castle_preserve_diagnostics(style, features)}
        san = board.san(move)
        expected += probability * static_after
        records.append(
            MoveRecord(
                move=move,
                san=san,
                uci=move.uci(),
                phi=phi,
                probability=probability,
                static_after=static_after,
                features=features,
                base_potential=base_phi,
                phase_potential=phase_phi,
            )
        )

    s = entropy(probabilities)
    return MoveLandscape(
        fen=board.fen(),
        turn="white" if board.turn == chess.WHITE else "black",
        beta=beta,
        records=records,
        entropy=s,
        effective_moves=effective_number(s),
        expected_value=expected,
    )
