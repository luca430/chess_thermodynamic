"""Player-dependent move potentials and probability measures."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping

import chess

from .evaluation import StaticEvaluator
from .features import BoardFeatureContext, board_context, move_feature_breakdown, phase_factors
from .metrics import effective_number, entropy

KAPPA = 1.0
STRATEGY_FEATURES = (
    "material",
    "center",
    "development",
    "castling",
    "king_safety",
    "king_pressure",
)


@dataclass(frozen=True)
class Style:
    material: float = 1.0
    center: float = 0.8
    development: float = 0.6
    castling: float = 0.6
    king_safety: float = 0.8
    king_pressure: float = 0.5

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def as_dict(self) -> Dict[str, float]:
        return {
            "material": self.material,
            "center": self.center,
            "development": self.development,
            "castling": self.castling,
            "king_safety": self.king_safety,
            "king_pressure": self.king_pressure,
        }


@dataclass(frozen=True)
class MoveRecord:
    move: chess.Move
    san: str
    uci: str
    phi: float
    probability: float
    static_after: float
    features: Dict[str, object]
    base_potential: float
    phase_potential: float
    static_components_after: Dict[str, float] | None = None
    static_terminal_after: bool = False
    static_terminal_reason_after: str | None = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "san": self.san,
            "uci": self.uci,
            "phi": self.phi,
            "probability": self.probability,
            "static_after": self.static_after,
            "static_components_after": self.static_components_after,
            "static_terminal_after": self.static_terminal_after,
            "static_terminal_reason_after": self.static_terminal_reason_after,
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
    kappa: float = KAPPA

    def as_dict(self) -> Dict[str, object]:
        return {
            "fen": self.fen,
            "turn": self.turn,
            "beta": self.beta,
            "entropy": self.entropy,
            "effective_moves": self.effective_moves,
            "expected_value": self.expected_value,
            "kappa": self.kappa,
            "temperature": temperature_from_beta(self.beta, self.kappa),
            "moves": [record.as_dict() for record in self.records],
        }


def beta_from_temperature(temperature: float, kappa: float = KAPPA) -> float:
    if kappa <= 0.0:
        raise ValueError("kappa must be greater than 0")
    if temperature <= 0.0:
        raise ValueError("temperature must be greater than 0")
    return 1.0 / (kappa * temperature)


def temperature_from_beta(beta: float, kappa: float = KAPPA) -> float:
    if kappa <= 0.0:
        raise ValueError("kappa must be greater than 0")
    if beta <= 0.0:
        raise ValueError("beta must be greater than 0")
    return 1.0 / (kappa * beta)


def effective_style_weights(style: Style, phase: float) -> Dict[str, float]:
    raw = style.as_dict()
    factors = phase_factors(phase)
    effective = {
        name: max(0.0, float(raw.get(name, 0.0))) * factors[name]
        for name in STRATEGY_FEATURES
    }
    total = sum(effective.values())
    if total <= 0.0:
        raise ValueError("at least one phase-effective style weight must be positive")
    return {name: value / total for name, value in effective.items()}


def strategy_weights(
    style: Style,
    features: Mapping[str, float] | None = None,
    *,
    phase_modulated: bool = True,
) -> Dict[str, float]:
    phase = 0.0 if features is None else float(features.get("game_phase", 0.0))
    if phase_modulated:
        return effective_style_weights(style, phase)
    raw = {name: max(0.0, float(style.as_dict().get(name, 0.0))) for name in STRATEGY_FEATURES}
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("at least one style weight must be positive")
    return {name: value / total for name, value in raw.items()}


def feature_energy_values(features: Mapping[str, float]) -> Dict[str, float]:
    """The six move features after fixed conversion to pawn-equivalent units."""

    return {name: float(features.get(name, 0.0)) for name in STRATEGY_FEATURES}


def potential_diagnostics(style: Style, features: Mapping[str, float]) -> Dict[str, object]:
    phase = float(features.get("game_phase", 0.0))
    raw = {name: max(0.0, float(style.as_dict().get(name, 0.0))) for name in STRATEGY_FEATURES}
    factors = phase_factors(phase)
    lambdas = effective_style_weights(style, phase)
    pawn_features = feature_energy_values(features)
    contributions = {
        name: lambdas[name] * pawn_features[name] for name in STRATEGY_FEATURES
    }
    return {
        "game_phase": phase,
        "raw_lambdas": raw,
        "phase_factors": factors,
        "effective_lambdas": lambdas,
        "contributions": contributions,
        "lambda_sum": sum(lambdas.values()),
        "phi": sum(contributions.values()),
    }


def potential_components(style: Style, features: Mapping[str, float]) -> tuple[float, float, float]:
    base_weights = strategy_weights(style, features, phase_modulated=False)
    pawn_features = feature_energy_values(features)
    base = sum(base_weights[name] * pawn_features[name] for name in STRATEGY_FEATURES)
    total = float(potential_diagnostics(style, features)["phi"])
    return base, total - base, total


def potential(style: Style, features: Mapping[str, float]) -> float:
    return potential_components(style, features)[2]


def boltzmann_probabilities(
    values: Iterable[float],
    *,
    beta: float | None = None,
    temperature: float | None = None,
    kappa: float = KAPPA,
) -> List[float]:
    if beta is None:
        if temperature is None:
            raise ValueError("either beta or temperature must be provided")
        beta = beta_from_temperature(temperature, kappa)
    elif beta <= 0.0:
        raise ValueError("beta must be greater than 0")
    if kappa <= 0.0:
        raise ValueError("kappa must be greater than 0")
    scaled = [beta * value for value in values]
    if not scaled:
        return []
    max_value = max(scaled)
    exp_values = [math.exp(value - max_value) for value in scaled]
    normalizer = sum(exp_values)
    if normalizer == 0.0:
        return [1.0 / len(exp_values)] * len(exp_values)
    return [value / normalizer for value in exp_values]


def stable_softmax(values: Iterable[float], beta: float) -> List[float]:
    return boltzmann_probabilities(values, beta=beta, kappa=KAPPA)


def _record_features(
    style: Style,
    pawn_features: Mapping[str, float],
    raw_features: Mapping[str, float],
    conversion_scales: Mapping[str, float],
    castling_raw_events: Mapping[str, float],
    castling_scales: Mapping[str, float],
) -> Dict[str, object]:
    diagnostics = potential_diagnostics(style, pawn_features)
    return {
        "raw_features": {name: float(raw_features[name]) for name in STRATEGY_FEATURES},
        "conversion_scales": {
            name: float(conversion_scales[name]) for name in STRATEGY_FEATURES
        },
        "pawn_features": {name: float(pawn_features[name]) for name in STRATEGY_FEATURES},
        "castling_raw_events": dict(castling_raw_events),
        "castling_scales": dict(castling_scales),
        "game_phase": diagnostics["game_phase"],
        "raw_lambdas": diagnostics["raw_lambdas"],
        "phase_factors": diagnostics["phase_factors"],
        "effective_lambdas": diagnostics["effective_lambdas"],
        "contributions": diagnostics["contributions"],
        "lambda_sum": diagnostics["lambda_sum"],
        "phi": diagnostics["phi"],
        **{name: float(pawn_features[name]) for name in STRATEGY_FEATURES},
    }


def move_distribution(
    board: chess.Board,
    style: Style,
    beta: float,
    evaluator: StaticEvaluator,
    before_context: BoardFeatureContext | None = None,
    context_provider: Callable[[chess.Board], BoardFeatureContext] | None = None,
    kappa: float = KAPPA,
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
            kappa=kappa,
        )

    after_boards = []
    after_contexts = []
    feature_breakdowns = []
    features_by_move = []
    for move in legal_moves:
        after = board.copy(stack=False)
        after.push(move)
        after_values = context_for(after)
        after_boards.append(after)
        after_contexts.append(after_values)
        breakdown = move_feature_breakdown(
            board,
            move,
            before_context=before,
            after_context=after_values,
            after_board=after,
        )
        feature_breakdowns.append(breakdown)
        pawn = breakdown["pawn"]
        features_by_move.append(
            {
                **{name: float(pawn[name]) for name in STRATEGY_FEATURES},
                "game_phase": float(breakdown["game_phase"]),
            }
        )

    components = [potential_components(style, features) for features in features_by_move]
    static_values = [
        evaluator.evaluate(after, context=after_values)
        for after, after_values in zip(after_boards, after_contexts)
    ]
    static_components = [
        evaluator.components(after, context=after_values)
        for after, after_values in zip(after_boards, after_contexts)
    ]
    terminal_reasons = [evaluator.terminal_reason(after) for after in after_boards]
    phis = [total for _, _, total in components]
    probabilities = boltzmann_probabilities(phis, beta=beta, kappa=kappa)

    records: List[MoveRecord] = []
    expected = 0.0
    for move, features, breakdown, (base_phi, phase_phi, _style_phi), phi, probability, static_after, components_after, terminal_reason in zip(
        legal_moves, features_by_move, feature_breakdowns, components, phis, probabilities, static_values, static_components, terminal_reasons
    ):
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
                features=_record_features(
                    style,
                    features,
                    breakdown["raw"],
                    breakdown["scales"],
                    breakdown["castling_raw_events"],
                    breakdown["castling_scales"],
                ),
                base_potential=base_phi,
                phase_potential=phase_phi,
                static_components_after=components_after,
                static_terminal_after=terminal_reason is not None,
                static_terminal_reason_after=terminal_reason,
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
        kappa=kappa,
    )
