from __future__ import annotations

import csv
import json
import importlib.util
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import chess
import pytest

import thermo_chess.features as feature_module
from thermo_chess.evaluation import EvaluationWeights, StaticEvaluator
from thermo_chess.features import (
    DEFAULT_SCALES,
    board_context,
    center_control,
    center_control_from_raw,
    center_control_raw,
    development_score_from_slots,
    development_score,
    development_slots,
    game_phase,
    king_freedom,
    king_freedom_from_safe_squares,
    king_safety,
    king_safety_from_raw,
    king_safety_raw,
    king_safe_squares,
    legal_mobility,
    legal_mobility_with_stats,
    mobility_score_from_count,
    material_balance,
    move_features,
    move_feature_breakdown,
    pawn_structure_counts,
    pawn_structure_score,
    pawn_structure_score_from_counts,
    unshielded_king_files,
)
from thermo_chess.measure import (
    KAPPA,
    STRATEGY_FEATURES,
    Style,
    beta_from_temperature,
    boltzmann_probabilities,
    consideration_indices,
    deepening_count_for_quantile,
    effective_support_size,
    effective_style_weights,
    move_distribution,
    potential,
    potential_components,
    potential_diagnostics,
    strategy_weights,
    temperature_from_beta,
)
from thermo_chess.metrics import effective_number, entropy
from thermo_chess.player import ThermoPlayer
from thermo_chess.search import AdaptiveExpectedValue, LandscapeObservation, SearchResult, adaptive_breadth, adaptive_cycle_limit, adaptive_recursive_plies_from_node, mass_preserving_expectation, requested_recursive_plies_for_cdepth, side_to_move_backup, validate_cdepth
from thermo_chess.simulation import (
    STRATEGY_NAMES,
    MatchConfig,
    _realized_cycle_payload,
    _terminal_result,
    _thermodynamic_transition_record,
    _termination_reason,
    captured_material,
    default_match_name,
    simulate_match,
    strategy_style,
)
from thermo_chess.thermodynamics import decompose_transition

_COVARIANCE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_static_covariances.py"
_COVARIANCE_SPEC = importlib.util.spec_from_file_location(
    "analyze_static_covariances", _COVARIANCE_SCRIPT
)
assert _COVARIANCE_SPEC is not None
_COVARIANCE_MODULE = importlib.util.module_from_spec(_COVARIANCE_SPEC)
assert _COVARIANCE_SPEC.loader is not None
_COVARIANCE_SPEC.loader.exec_module(_COVARIANCE_MODULE)
COMPONENT_ORDER = _COVARIANCE_MODULE.COMPONENT_ORDER
analyze_game = _COVARIANCE_MODULE.analyze_game
candidate_covariance_record = _COVARIANCE_MODULE.candidate_covariance_record
normalize_probabilities = _COVARIANCE_MODULE.normalize_probabilities
weighted_mean_and_covariance = _COVARIANCE_MODULE.weighted_mean_and_covariance

_DIVERGENCE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_deep_search_divergence.py"
_DIVERGENCE_SPEC = importlib.util.spec_from_file_location(
    "analyze_deep_search_divergence", _DIVERGENCE_SCRIPT
)
assert _DIVERGENCE_SPEC is not None
_DIVERGENCE_MODULE = importlib.util.module_from_spec(_DIVERGENCE_SPEC)
assert _DIVERGENCE_SPEC.loader is not None
_DIVERGENCE_SPEC.loader.exec_module(_DIVERGENCE_MODULE)
DivergenceAnalysisError = _DIVERGENCE_MODULE.DivergenceAnalysisError
divergence_rows = _DIVERGENCE_MODULE.divergence_rows
rank_by_abs_divergence = _DIVERGENCE_MODULE.rank_by_abs_divergence
rows_for_ply = _DIVERGENCE_MODULE.rows_for_ply


def _quiet_style(**overrides: float) -> Style:
    values = {name: 0.0 for name in STRATEGY_FEATURES}
    values.update(overrides)
    if all(value == 0.0 for value in values.values()):
        values["material"] = 1.0
    return Style(**values)


def _move_record(board: chess.Board, style: Style, uci: str) -> dict[str, object]:
    landscape = move_distribution(board, style, beta=1.0, evaluator=StaticEvaluator())
    for record in landscape.records:
        if record.uci == uci:
            return record.as_dict()
    raise AssertionError(f"move {uci} not found")


def _old_legal_mobility(board: chess.Board, color: chess.Color) -> int:
    probe = board.copy(stack=False)
    probe.turn = color
    probe.ep_square = None
    if probe.is_checkmate() or probe.is_stalemate() or probe.is_insufficient_material():
        return 0
    return probe.legal_moves.count()


def test_starting_board_static_symmetry_and_phase() -> None:
    board = chess.Board()
    context = board_context(board)
    evaluator = StaticEvaluator()

    assert context.material_balance == pytest.approx(0.0)
    assert context.pawn_structure_by_side[chess.WHITE] - context.pawn_structure_by_side[chess.BLACK] == pytest.approx(0.0)
    assert context.center_by_side[chess.WHITE] - context.center_by_side[chess.BLACK] == pytest.approx(0.0)
    assert evaluator.evaluate(board, context=context) == pytest.approx(0.0)
    assert game_phase(board) == pytest.approx(0.0)


def test_material_values_are_pawn_units_and_promotion_is_material_only() -> None:
    white_up_pawn = chess.Board("4k3/8/8/8/8/8/8/4KP2 w - - 0 1")
    white_up_queen = chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")
    promotion = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    features = move_features(promotion, chess.Move.from_uci("a7a8q"))

    assert material_balance(white_up_pawn) == pytest.approx(1.0)
    assert material_balance(white_up_queen) == pytest.approx(9.0)
    assert features["material"] == pytest.approx(8.0)
    assert "promotion" not in features


def test_pawn_structure_counts_and_scores() -> None:
    white = chess.Board("4k3/8/8/8/8/8/2P1P3/4K3 w - - 0 1")
    doubled = chess.Board("4k3/8/8/8/8/2P5/2P5/4K3 w - - 0 1")
    passed_white = chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")
    passed_black = chess.Board("4k3/8/8/8/8/3p4/8/4K3 b - - 0 1")
    connected = chess.Board("4k3/8/8/8/8/2PP4/8/4K3 w - - 0 1")

    assert pawn_structure_counts(white, chess.WHITE).isolated == 2
    assert pawn_structure_counts(doubled, chess.WHITE).doubled == 1
    assert pawn_structure_counts(passed_white, chess.WHITE).passed == 1
    assert pawn_structure_counts(passed_black, chess.BLACK).passed == 1
    assert pawn_structure_counts(connected, chess.WHITE).connected == 2
    assert pawn_structure_score(connected, chess.WHITE) == pytest.approx(0.30 * 2 + 0.10 * 2)


def test_mobility_counts_and_pawn_conversion() -> None:
    board = chess.Board()
    context = board_context(board)
    assert legal_mobility(board, chess.WHITE) == 20
    assert legal_mobility(board, chess.BLACK) == 20
    assert context.mobility_score_by_side[chess.WHITE] == pytest.approx(20 * 0.03)


def test_center_control_values_and_no_duplicate_attack_reward() -> None:
    core = chess.Board("4k3/8/8/8/8/2P5/8/4K3 w - - 0 1")
    extended = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    duplicate_one = chess.Board("4k3/8/8/8/8/5N2/8/4K3 w - - 0 1")
    duplicate_two = chess.Board("4k3/8/8/8/8/1N3N2/8/4K3 w - - 0 1")

    assert center_control(core, chess.WHITE) == pytest.approx(0.10)
    assert center_control_raw(core, chess.WHITE) == pytest.approx(1.0)
    assert center_control(extended, chess.WHITE) == pytest.approx(2 * 0.035)
    assert len(duplicate_two.attackers(chess.WHITE, chess.D4)) == 2
    assert center_control(duplicate_two, chess.WHITE) == pytest.approx(2 * 0.10 + 0.035)


def test_king_safety_components() -> None:
    base = chess.Board("8/8/8/8/8/3PPP2/8/4K2k w - - 0 1")
    shield = chess.Board("8/8/8/8/8/8/3PPP2/4K2k w - - 0 1")
    attacked = chess.Board("8/8/8/8/8/8/4p3/4K2k w - - 0 1")

    assert king_safety(shield, chess.WHITE) - king_safety(base, chess.WHITE) == pytest.approx(3 * 0.12)
    assert king_safety(attacked, chess.WHITE) < king_safety(base, chess.WHITE)
    unshielded = chess.Board("8/8/8/8/8/8/8/4K2k w - - 0 1")
    assert unshielded_king_files(unshielded, chess.WHITE) == 3
    assert king_safety(unshielded, chess.WHITE) == pytest.approx(-3 * 0.15)


def test_development_only_original_minor_slots() -> None:
    board = chess.Board()
    knight = board.copy(stack=False)
    knight.push(chess.Move.from_uci("g1f3"))
    pawn = board.copy(stack=False)
    pawn.push(chess.Move.from_uci("e2e4"))

    assert development_score(knight, chess.WHITE) - development_score(board, chess.WHITE) == pytest.approx(0.15)
    assert development_slots(knight, chess.WHITE) - development_slots(board, chess.WHITE) == 1
    assert development_score(pawn, chess.WHITE) == pytest.approx(development_score(board, chess.WHITE))


def test_castling_transition_feature() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert move_features(board, chess.Move.from_uci("e1g1"))["castling"] == pytest.approx(0.40)
    assert move_features(board, chess.Move.from_uci("e1f1"))["castling"] == pytest.approx(-0.40)
    assert move_features(board, chess.Move.from_uci("h1h2"))["castling"] == pytest.approx(-0.20)


def test_king_pressure_uses_local_freedom_not_check_feature() -> None:
    board = chess.Board("7k/8/5K2/8/8/8/6Q1/8 w - - 0 1")
    features = move_features(board, chess.Move.from_uci("g2g6"))
    after = board.copy(stack=False)
    after.push(chess.Move.from_uci("g2g6"))

    assert features["king_pressure"] == pytest.approx(
        king_freedom(board, chess.BLACK, chess.WHITE)
        - king_freedom(after, chess.BLACK, chess.WHITE)
    )
    assert features["king_pressure"] == pytest.approx(
        DEFAULT_SCALES.king_freedom
        * (
            king_safe_squares(board, chess.BLACK, chess.WHITE)
            - king_safe_squares(after, chess.BLACK, chess.WHITE)
        )
    )
    assert features["king_pressure"] > 0.0
    assert "check" not in features
    assert "mate" not in features


def test_raw_to_scaled_feature_helpers_match_public_helpers() -> None:
    boards = [
        chess.Board(),
        chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 8"),
        chess.Board("8/8/8/8/8/3PPP2/8/4K2k w - - 0 1"),
    ]

    for board in boards:
        for color in (chess.WHITE, chess.BLACK):
            pawn_counts = pawn_structure_counts(board, color)
            mobility = legal_mobility(board, color)
            center_raw = center_control_raw(board, color)
            safety_raw = king_safety_raw(board, color)
            slots = development_slots(board, color)
            safe_squares = king_safe_squares(board, color, not color)

            assert pawn_structure_score_from_counts(pawn_counts) == pytest.approx(
                pawn_structure_score(board, color)
            )
            assert mobility_score_from_count(mobility) == pytest.approx(0.03 * mobility)
            assert center_control_from_raw(center_raw) == pytest.approx(center_control(board, color))
            assert king_safety_from_raw(safety_raw) == pytest.approx(king_safety(board, color))
            assert development_score_from_slots(slots) == pytest.approx(development_score(board, color))
            assert king_freedom_from_safe_squares(safe_squares) == pytest.approx(
                king_freedom(board, color, not color)
            )


def test_board_context_fields_match_public_helpers_for_representative_positions() -> None:
    boards = [
        chess.Board(),
        chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 8"),
        chess.Board("8/8/8/2pPp3/2P1P3/8/8/4K2k w - e6 0 1"),
        chess.Board("r4rk1/ppp2ppp/2n5/8/8/2N2N2/PPP2PPP/R4RK1 w - - 0 1"),
        chess.Board("8/8/8/8/8/8/4p3/4K2k w - - 0 1"),
        chess.Board("8/8/8/8/8/8/P7/K1k5 w - - 0 1"),
    ]

    for board in boards:
        context = board_context(board)
        for color in (chess.WHITE, chess.BLACK):
            assert context.pawn_counts_by_side[color] == pawn_structure_counts(board, color)
            assert context.pawn_structure_by_side[color] == pytest.approx(pawn_structure_score(board, color))
            assert context.mobility_by_side[color] == legal_mobility(board, color)
            assert context.mobility_score_by_side[color] == pytest.approx(
                mobility_score_from_count(context.mobility_by_side[color])
            )
            assert context.center_raw_by_side[color] == pytest.approx(center_control_raw(board, color))
            assert context.center_by_side[color] == pytest.approx(center_control(board, color))
            assert context.king_safety_raw_by_side[color] == pytest.approx(king_safety_raw(board, color))
            assert context.king_safety_by_side[color] == pytest.approx(king_safety(board, color))
            assert context.development_slots_by_side[color] == development_slots(board, color)
            assert context.development_by_side[color] == pytest.approx(development_score(board, color))
            assert context.king_safe_squares_by_side[color] == king_safe_squares(board, color, not color)
            assert context.king_freedom_by_side[color] == pytest.approx(king_freedom(board, color, not color))


def test_board_context_computes_primitives_once_per_side(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {
        "pawn_structure_counts": 0,
        "legal_mobility_with_stats": 0,
        "center_control_raw": 0,
        "king_safety_raw": 0,
        "development_slots": 0,
        "king_safe_squares": 0,
    }
    originals = {name: getattr(feature_module, name) for name in counts}

    def counted(name: str):
        original = originals[name]

        def wrapper(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        return wrapper

    for name in counts:
        monkeypatch.setattr(feature_module, name, counted(name))

    board_context(chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 8"))

    assert counts == {
        "pawn_structure_counts": 2,
        "legal_mobility_with_stats": 2,
        "center_control_raw": 2,
        "king_safety_raw": 2,
        "development_slots": 2,
        "king_safe_squares": 2,
    }


def test_phase_depends_on_non_pawn_material_only() -> None:
    start = chess.Board()
    no_pawns = chess.Board("rnbqkbnr/8/8/8/8/8/8/RNBQKBNR w KQkq - 0 1")
    kings = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    no_queens = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    one_knight_removed = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1")
    one_rook_removed = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kkq - 0 1")

    assert game_phase(start) == pytest.approx(0.0)
    assert game_phase(no_pawns) == pytest.approx(0.0)
    assert game_phase(one_knight_removed) == pytest.approx(3.0 / 62.0)
    assert game_phase(one_rook_removed) == pytest.approx(5.0 / 62.0)
    assert game_phase(no_queens) == pytest.approx(1.0 - 44.0 / 62.0)
    assert game_phase(kings) == pytest.approx(1.0)


def test_lambda_normalization_and_phase_dependence() -> None:
    style = Style(material=1.0, center=1.0, development=1.0, castling=1.0, king_safety=1.0, king_pressure=1.0)
    early = effective_style_weights(style, 0.0)
    late = effective_style_weights(style, 0.9)

    for weights in (early, late, strategy_weights(style, {"game_phase": 0.45})):
        assert all(value >= 0.0 for value in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0)
    assert late["development"] < early["development"]
    assert late["center"] < early["center"]
    assert late["castling"] < early["castling"]
    assert late["king_safety"] < early["king_safety"]
    assert late["material"] > early["material"]
    assert late["king_pressure"] > early["king_pressure"]
    with pytest.raises(ValueError):
        effective_style_weights(Style(material=0.0, center=0.0, development=0.0, castling=0.0, king_safety=0.0, king_pressure=0.0), 0.0)


def test_potential_reconstructs_from_lambdas_and_features() -> None:
    board = chess.Board()
    features = move_features(board, chess.Move.from_uci("g1f3"))
    style = Style(material=1.0, center=2.0, development=3.0, castling=4.0, king_safety=5.0, king_pressure=6.0)
    diagnostics = potential_diagnostics(style, features)
    reconstructed = sum(
        diagnostics["effective_lambdas"][name] * features[name]
        for name in STRATEGY_FEATURES
    )

    assert potential(style, features) == pytest.approx(reconstructed)
    assert potential_components(style, features)[2] == pytest.approx(reconstructed)
    assert diagnostics["phi"] == pytest.approx(reconstructed)


def test_raw_features_convert_to_pawn_features() -> None:
    board = chess.Board()
    breakdown = move_feature_breakdown(board, chess.Move.from_uci("g1f3"))
    raw = breakdown["raw"]
    scales = breakdown["scales"]
    pawn = breakdown["pawn"]

    for name in STRATEGY_FEATURES:
        if name == "castling":
            events = breakdown["castling_raw_events"]
            castling_scales = breakdown["castling_scales"]
            expected = (
                castling_scales["castle"] * events["castle"]
                + castling_scales["rights"] * events["rights"]
            )
        else:
            expected = raw[name] * scales[name]
        assert pawn[name] == pytest.approx(expected)


def test_move_record_serializes_model_diagnostics() -> None:
    record = _move_record(chess.Board(), Style(), "g1f3")
    features = record["features"]

    assert set(features["raw_features"]) == set(STRATEGY_FEATURES)
    assert set(features["conversion_scales"]) == set(STRATEGY_FEATURES)
    assert set(features["pawn_features"]) == set(STRATEGY_FEATURES)
    assert set(features["raw_lambdas"]) == set(STRATEGY_FEATURES)
    assert set(features["phase_factors"]) == set(STRATEGY_FEATURES)
    assert set(features["effective_lambdas"]) == set(STRATEGY_FEATURES)
    assert set(features["contributions"]) == set(STRATEGY_FEATURES)
    assert features["lambda_sum"] == pytest.approx(1.0)
    assert features["phi"] == pytest.approx(record["total_potential"])
    assert features["phi"] == pytest.approx(
        sum(features["effective_lambdas"][name] * features["pawn_features"][name] for name in STRATEGY_FEATURES)
    )


def test_no_generic_extra_or_objective_weight_escape_hatch() -> None:
    with pytest.raises(TypeError):
        Style(extra={"material": 1.0})
    with pytest.raises(TypeError):
        EvaluationWeights(extra={"material": 1.0})
    with pytest.raises(TypeError):
        move_distribution(
            chess.Board(),
            Style(),
            1.0,
            StaticEvaluator(),
            value_objective_weight=1.0,
        )


def test_hypothetical_mobility_does_not_inherit_en_passant_right() -> None:
    board = chess.Board()
    for san in ("e4", "a5", "e5", "d5"):
        board.push_san(san)
    assert board.ep_square == chess.D6
    without_ep = board.copy(stack=False)
    without_ep.ep_square = None
    assert legal_mobility(board, chess.WHITE) == without_ep.legal_moves.count()


def test_legal_mobility_matches_old_probe_semantics_for_representative_positions() -> None:
    positions = [
        chess.Board(),
        chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 8"),
        chess.Board("4k3/8/8/8/8/3r4/4N3/4K3 w - - 0 1"),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1"),
        chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"),
        chess.Board("8/P7/8/8/8/8/8/K1k5 w - - 0 1"),
        chess.Board("8/8/8/8/8/8/P7/K1k5 w - - 0 1"),
        chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"),
        chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
    ]

    for board in positions:
        for color in (chess.WHITE, chess.BLACK):
            assert legal_mobility(board, color) == _old_legal_mobility(board, color)


def test_legal_mobility_uses_direct_path_for_actual_side_without_ep() -> None:
    board = chess.Board()
    stats: dict[str, int] = {}

    value = legal_mobility_with_stats(board, board.turn, stats)

    assert value == board.legal_moves.count()
    assert stats["legal_mobility_requests"] == 1
    assert stats["legal_mobility_direct_counts"] == 1
    assert stats.get("legal_mobility_probe_counts", 0) == 0
    assert stats.get("legal_mobility_board_copies", 0) == 0


def test_legal_mobility_probe_path_preserves_opposite_side_and_ep_semantics() -> None:
    board = chess.Board()
    for san in ("e4", "a5", "e5", "d5"):
        board.push_san(san)
    before_fen = board.fen()
    stats: dict[str, int] = {}

    actual = legal_mobility_with_stats(board, board.turn, stats)
    opposite = legal_mobility_with_stats(board, not board.turn, stats)

    assert board.fen() == before_fen
    assert actual == _old_legal_mobility(board, board.turn)
    assert opposite == _old_legal_mobility(board, not board.turn)
    assert stats["legal_mobility_requests"] == 2
    assert stats["legal_mobility_probe_counts"] == 2
    assert stats["legal_mobility_board_copies"] == 2
    assert stats.get("legal_mobility_direct_counts", 0) == 0


def test_board_context_mobility_uses_one_probe_copy_without_ep() -> None:
    stats: dict[str, int] = {}
    context = board_context(chess.Board(), stats=stats)

    assert context.mobility_by_side[chess.WHITE] == 20
    assert context.mobility_by_side[chess.BLACK] == 20
    assert stats["legal_mobility_requests"] == 2
    assert stats["legal_mobility_direct_counts"] == 1
    assert stats["legal_mobility_probe_counts"] == 1
    assert stats["legal_mobility_board_copies"] == 1


def test_probabilities_sum_and_beta_changes_concentration_not_phi() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    low = move_distribution(board, Style(center=4.0), 0.5, evaluator)
    high = move_distribution(board, Style(center=4.0), 20.0, evaluator)

    assert sum(record.probability for record in low.records) == pytest.approx(1.0)
    assert sum(record.probability for record in high.records) == pytest.approx(1.0)
    assert max(record.probability for record in high.records) > max(record.probability for record in low.records)
    assert [record.phi for record in high.records] == pytest.approx([record.phi for record in low.records])
    assert sum(boltzmann_probabilities([0.0, 100.0, 200.0], beta=100.0)) == pytest.approx(1.0)


def test_material_imbalance_changes_static_sign_and_terminal_positions() -> None:
    evaluator = StaticEvaluator()
    white_up_queen = chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")
    black_up_queen = chess.Board("4kq2/8/8/8/8/8/8/4K3 w - - 0 1")
    black_mated = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")

    assert evaluator.evaluate(white_up_queen) > 0
    assert evaluator.evaluate(black_up_queen) < 0
    assert evaluator.evaluate(black_mated) > 9000
    assert evaluator.evaluate(stalemate) == 0.0


def test_static_component_decomposition_matches_scalar_evaluator() -> None:
    board = chess.Board()
    for san in ("e4", "c5", "Nf3", "d6", "Bb5+"):
        board.push_san(san)
    weights = EvaluationWeights(
        material=1.2,
        pawn_structure=0.7,
        mobility=1.4,
        center=0.8,
        king_safety=1.1,
    )
    evaluator = StaticEvaluator(weights)
    components = evaluator.components(board)

    assert components is not None
    assert tuple(components) == COMPONENT_ORDER
    reconstructed = sum(weights.as_dict()[name] * components[name] for name in COMPONENT_ORDER)
    assert reconstructed == pytest.approx(evaluator.evaluate(board))


def test_combined_static_evaluator_matches_scalar_and_components() -> None:
    positions = [
        chess.Board(),
        chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 8"),
        chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1"),
    ]
    evaluator = StaticEvaluator(
        EvaluationWeights(
            material=1.2,
            pawn_structure=0.7,
            mobility=1.4,
            center=0.8,
            king_safety=1.1,
        )
    )

    for board in positions:
        context = board_context(board)
        value, components = evaluator.evaluate_with_components(board, context=context)
        assert value == pytest.approx(evaluator.evaluate(board, context=context))
        assert components == evaluator.components(board, context=context)


def test_combined_static_evaluator_preserves_terminal_values() -> None:
    evaluator = StaticEvaluator()
    boards = [
        chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"),
        chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
        chess.Board("8/8/8/8/8/8/8/K1k5 w - - 0 1"),
    ]

    for board in boards:
        value, components = evaluator.evaluate_with_components(board)
        assert value == pytest.approx(evaluator.evaluate(board))
        assert components is None


def test_move_distribution_computes_static_components_once_per_considered_child() -> None:
    class ComponentCountingEvaluator(StaticEvaluator):
        def __init__(self) -> None:
            super().__init__()
            self.component_calls = 0

        def components(self, board: chess.Board, context=None):  # type: ignore[override]
            self.component_calls += 1
            return super().components(board, context=context)

    board = chess.Board()
    evaluator = ComponentCountingEvaluator()
    landscape = move_distribution(board, Style(center=1.5), 2.0, evaluator)

    assert evaluator.component_calls == landscape.consideration_count
    assert landscape.consideration_count == sum(
        1 for record in landscape.records if record.in_consideration_set
    )


def test_static_components_are_null_for_terminal_positions() -> None:
    evaluator = StaticEvaluator()
    black_mated = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")

    assert evaluator.components(black_mated) is None
    assert evaluator.terminal_reason(black_mated) == "checkmate"
    assert evaluator.evaluate(black_mated) > 9000


def test_static_evaluation_weights_are_dimensionless() -> None:
    evaluator = StaticEvaluator(
        EvaluationWeights(
            material=0.0,
            pawn_structure=0.0,
            mobility=1.0,
            center=0.0,
            king_safety=0.0,
        )
    )
    board = chess.Board()
    assert evaluator.evaluate(board) == pytest.approx(0.0)


def test_kappa_beta_temperature_and_entropy_helpers() -> None:
    assert KAPPA == pytest.approx(1.0)
    assert beta_from_temperature(0.25, KAPPA) == pytest.approx(4.0)
    assert temperature_from_beta(4.0, KAPPA) == pytest.approx(0.25)
    assert entropy([0.2, 0.3, 0.5]) >= 0.0
    assert effective_number(entropy([0.25] * 4)) == pytest.approx(4.0)


def test_effective_support_uses_ceil_neff_and_top_raw_probabilities() -> None:
    probabilities = [0.60, 0.20, 0.10, 0.05, 0.05]
    s = -sum(probability * math.log(probability) for probability in probabilities)
    n_eff = math.exp(s)

    assert entropy(probabilities) == pytest.approx(s)
    assert effective_number(s) == pytest.approx(n_eff)
    assert effective_support_size(probabilities) == math.ceil(n_eff)
    assert consideration_indices(probabilities) == [0, 1, 2, 3]


def test_considered_probabilities_renormalize_and_quantile_deepens_subset() -> None:
    probabilities = [0.60, 0.20, 0.10, 0.05, 0.05]
    considered = consideration_indices(probabilities)
    mass = sum(probabilities[index] for index in considered)
    considered_probabilities = [probabilities[index] / mass for index in considered]

    assert sum(considered_probabilities) == pytest.approx(1.0)
    assert considered_probabilities == pytest.approx([12 / 19, 4 / 19, 2 / 19, 1 / 19])
    assert deepening_count_for_quantile(considered_probabilities, 0.8) == 2
    with pytest.raises(ValueError):
        deepening_count_for_quantile(considered_probabilities, 0.0)


@pytest.mark.parametrize(
    ("effective_moves", "adaptive_c", "legal_moves", "expected"),
    [(20.0, 0.3, 30, 6), (3.0, 0.3, 20, 1), (20.0, 2.0, 7, 7), (1.0, 0.01, 5, 1)],
)
def test_adaptive_breadth_rule(effective_moves: float, adaptive_c: float, legal_moves: int, expected: int) -> None:
    assert adaptive_breadth(effective_moves, adaptive_c, legal_moves) == expected


@pytest.mark.parametrize(
    ("consideration_count", "cdepth", "expected"),
    [
        (100, 1, 1),
        (16, 1, 1),
        (15, 1, 1),
        (6, 1, 1),
        (5, 1, 1),
        (2, 1, 1),
        (16, 2, 1),
        (15, 2, 2),
        (6, 2, 2),
        (5, 2, 2),
        (2, 2, 2),
        (16, 3, 1),
        (15, 3, 2),
        (6, 3, 2),
        (5, 3, 3),
        (2, 3, 3),
    ],
)
def test_adaptive_cycle_limit_thresholds(
    consideration_count: int,
    cdepth: int,
    expected: int,
) -> None:
    assert adaptive_cycle_limit(consideration_count, cdepth) == expected


@pytest.mark.parametrize("cdepth", [1, 2, 3])
def test_adaptive_cycle_limit_treats_single_deepening_branch_as_forced(cdepth: int) -> None:
    assert adaptive_cycle_limit(1, cdepth) is None


@pytest.mark.parametrize(
    ("cycle_limit", "level", "expected"),
    [
        (1, 1, 1),
        (1, 2, 2),
        (1, 3, 1),
        (1, 4, 2),
        (2, 1, 3),
        (2, 2, 4),
        (2, 3, 3),
        (2, 4, 4),
        (3, 1, 5),
        (3, 2, 6),
        (3, 3, 5),
        (3, 4, 6),
    ],
)
def test_adaptive_recursive_plies_from_node_uses_phase_not_absolute_level(
    cycle_limit: int,
    level: int,
    expected: int,
) -> None:
    assert adaptive_recursive_plies_from_node(cycle_limit, level) == expected


def test_adaptive_recursive_plies_from_node_preserves_same_phase_allowance() -> None:
    for cycle_limit in (1, 2, 3):
        assert adaptive_recursive_plies_from_node(cycle_limit, 1) == adaptive_recursive_plies_from_node(cycle_limit, 3)
        assert adaptive_recursive_plies_from_node(cycle_limit, 2) == adaptive_recursive_plies_from_node(cycle_limit, 4)


def test_mass_preserving_expectation_synthetic_limits_and_partial_case() -> None:
    no_deep = (
        LandscapeObservation(chess.Move.from_uci("a2a3"), "a2a3", 0.2, 1.0, False, 0),
        LandscapeObservation(chess.Move.from_uci("b2b3"), "b2b3", 0.8, 3.0, False, 0),
    )
    all_deep = (
        LandscapeObservation(chess.Move.from_uci("a2a3"), "a2a3", 0.2, 5.0, True, 1),
        LandscapeObservation(chess.Move.from_uci("b2b3"), "b2b3", 0.8, 7.0, True, 1),
    )
    partial = (
        LandscapeObservation(chess.Move.from_uci("a2a3"), "a2a3", 0.2, 5.0, True, 1),
        LandscapeObservation(chess.Move.from_uci("b2b3"), "b2b3", 0.3, 11.0, True, 1),
        LandscapeObservation(chess.Move.from_uci("c2c3"), "c2c3", 0.5, -2.0, False, 0),
    )

    assert mass_preserving_expectation(no_deep) == pytest.approx(0.2 * 1.0 + 0.8 * 3.0)
    assert mass_preserving_expectation(all_deep) == pytest.approx(0.2 * 5.0 + 0.8 * 7.0)
    assert mass_preserving_expectation(partial) == pytest.approx(0.2 * 5.0 + 0.3 * 11.0 + 0.5 * -2.0)


def test_search_result_serialization_and_parallel_match() -> None:
    board = chess.Board("7k/8/8/8/8/8/6K1/7R w - - 0 1")
    evaluator = StaticEvaluator()
    result = AdaptiveExpectedValue(Style(), 1.0, evaluator, cdepth=2).search_result(board, player="white", side="white")
    restored = SearchResult.from_dict(result.as_dict())
    assert restored.U == pytest.approx(result.U)
    assert [move.uci for move in restored.moves] == [move.uci for move in result.moves]
    assert restored.evaluation_weights == result.evaluation_weights

    serial = AdaptiveExpectedValue(Style(), 1.0, evaluator, cdepth=2, search_workers=1).search_result(board)
    parallel = AdaptiveExpectedValue(Style(), 1.0, evaluator, cdepth=2, search_workers=2, parallel_min_branches=1).search_result(board)
    assert serial.U == pytest.approx(parallel.U)
    serial_selected = max((move for move in serial.moves if move.g_tilde is not None), key=lambda move: move.g_tilde)
    parallel_selected = max((move for move in parallel.moves if move.g_tilde is not None), key=lambda move: move.g_tilde)
    assert serial_selected.uci == parallel_selected.uci
    assert serial_selected.g_tilde == pytest.approx(parallel_selected.g_tilde)
    assert serial_selected.thermo_g_tilde == pytest.approx(parallel_selected.thermo_g_tilde)
    assert serial_selected.q_tilde == pytest.approx(parallel_selected.q_tilde)
    assert serial_selected.w_tilde == pytest.approx(parallel_selected.w_tilde)
    assert serial_selected.a_tilde == pytest.approx(parallel_selected.a_tilde)
    assert serial_selected.deepened_cycles == parallel_selected.deepened_cycles
    assert serial_selected.principal_variation == parallel_selected.principal_variation
    assert serial_selected.search_endpoint_fen == parallel_selected.search_endpoint_fen
    assert serial_selected.thermodynamic_endpoint_fen == parallel_selected.thermodynamic_endpoint_fen


def test_broad_cdepth_one_root_keeps_one_adaptive_cycle() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    search = AdaptiveExpectedValue(Style(), 1.0, evaluator, cdepth=1, deepening_quantile=0.8)
    result = search.search_result(board)

    considered = [move for move in result.moves if move.in_consideration_set]
    assert result.K > 15
    assert result.diagnostics["root_adaptive_cycle_limit"] == 1
    assert result.diagnostics["root_adaptive_recursive_plies"] == 1
    assert result.diagnostics["deepened_move_count"] > 0
    assert result.selected_uci
    assert considered
    assert all(move.g_tilde is not None for move in considered)
    assert any(move.deepened_cycles == 1 for move in considered if move.selected_for_refinement)


def test_broad_branching_clips_all_configured_depths_to_one_cycle() -> None:
    board = chess.Board()
    for cdepth in (1, 2, 3):
        result = AdaptiveExpectedValue(
            Style(),
            1.0,
            StaticEvaluator(),
            cdepth=cdepth,
            deepening_quantile=1.0,
        ).search_result(board)

        assert result.K > 15
        assert result.diagnostics["root_adaptive_cycle_limit"] == 1
        assert result.diagnostics["root_adaptive_recursive_plies"] == 1
        assert result.selected_uci
        assert max(move.deepened_cycles for move in result.moves if move.in_consideration_set) <= 1


def test_medium_branching_caps_cdepth_three_at_two_cycles() -> None:
    board = chess.Board("8/8/8/8/8/8/PPP5/K2k4 w - - 0 1")
    result = AdaptiveExpectedValue(
        Style(),
        1.0,
        StaticEvaluator(),
        cdepth=3,
        deepening_quantile=1.0,
    ).search_result(board)

    assert 6 <= result.K <= 15
    assert result.diagnostics["root_adaptive_cycle_limit"] == 2
    assert result.diagnostics["root_adaptive_recursive_plies"] == 3
    assert result.selected_uci
    assert max(move.deepened_cycles for move in result.moves if move.in_consideration_set) <= 2


def test_narrow_branching_keeps_full_depth_available() -> None:
    board = chess.Board("8/8/8/8/8/8/P7/K1k5 w - - 0 1")
    result = AdaptiveExpectedValue(
        Style(),
        1.0,
        StaticEvaluator(),
        cdepth=2,
        deepening_quantile=1.0,
    ).search_result(board)

    assert result.K <= 5
    assert result.diagnostics["root_adaptive_cycle_limit"] == 2
    assert result.diagnostics["root_adaptive_recursive_plies"] == 3
    assert result.selected_uci
    assert max(move.deepened_cycles for move in result.moves if move.in_consideration_set) <= 2
    assert any(move.recursive_plies_used > 1 for move in result.moves if move.in_consideration_set)


class _ForcedConsiderationSearch(AdaptiveExpectedValue):
    def __init__(self, *args: object, forced_counts: dict[str, int], **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._forced_counts = forced_counts

    def landscape(self, board: chess.Board, key: object | None = None):  # type: ignore[override]
        landscape = super().landscape(board, key)  # type: ignore[arg-type]
        forced = self._forced_counts.get(board.fen())
        if forced is None and "__non_root__" in self._forced_counts:
            forced = self._forced_counts["__non_root__"]
        return landscape if forced is None else replace(landscape, consideration_count=forced)

    def _new_branch_context(self) -> "_ForcedConsiderationSearch":
        return _ForcedConsiderationSearch(
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
            forced_counts=self._forced_counts,
        )


class _ControlledConsiderationSearch(AdaptiveExpectedValue):
    def __init__(self, *args: object, considered_count: int, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._considered_count = considered_count

    def landscape(self, board: chess.Board, key: object | None = None):  # type: ignore[override]
        landscape = super().landscape(board, key)  # type: ignore[arg-type]
        records = list(landscape.records)
        considered_count = min(self._considered_count, len(records))
        probability = 1.0 / considered_count if considered_count else 0.0
        controlled = tuple(
            replace(
                record,
                probability=probability if index < considered_count else 0.0,
                considered_probability=probability if index < considered_count else 0.0,
                in_consideration_set=index < considered_count,
                consideration_rank=index + 1 if index < considered_count else None,
            )
            for index, record in enumerate(records)
        )
        return replace(
            landscape,
            records=list(controlled),
            consideration_count=considered_count,
            consideration_probability_mass_raw=1.0,
        )

    def _new_branch_context(self) -> "_ControlledConsiderationSearch":
        return _ControlledConsiderationSearch(
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
            considered_count=self._considered_count,
        )


def test_adaptive_depth_removed_by_ancestor_cannot_be_restored() -> None:
    board = chess.Board()
    result = AdaptiveExpectedValue(
        Style(),
        1.0,
        StaticEvaluator(),
        cdepth=3,
        deepening_quantile=0.8,
    ).search_result(board)

    assert result.diagnostics["root_adaptive_cycle_limit"] == 1
    assert result.diagnostics["root_adaptive_recursive_plies"] == 1
    assert max(move.deepened_cycles for move in result.moves if move.in_consideration_set) <= 1


def test_deeper_node_can_reduce_inherited_depth_to_two_cycles() -> None:
    board = chess.Board("8/8/8/8/8/8/P1k5/K7 w - - 0 1")
    result = AdaptiveExpectedValue(
        Style(),
        1.0,
        StaticEvaluator(),
        cdepth=3,
        deepening_quantile=1.0,
    ).search_result(board)

    assert result.diagnostics["root_adaptive_cycle_limit"] == 3
    assert result.diagnostics["root_adaptive_recursive_plies"] == 5
    assert result.selected_uci
    assert max(move.deepened_cycles for move in result.moves if move.in_consideration_set) <= 2


def test_depth_uses_quantile_count_before_consideration_count() -> None:
    board = chess.Board("8/8/8/8/8/8/RP6/K2k4 w - - 0 1")
    concentrated = AdaptiveExpectedValue(
        Style(center=10.0, development=0.1, material=0.1, castling=0.1, king_safety=0.1, king_pressure=0.1),
        16.0,
        StaticEvaluator(),
        cdepth=3,
        deepening_quantile=0.8,
    ).search_result(board)
    diffuse = AdaptiveExpectedValue(
        Style(),
        1.0,
        StaticEvaluator(),
        cdepth=3,
        deepening_quantile=0.8,
    ).search_result(board)

    assert concentrated.diagnostics["root_consideration_count"] == 5
    assert concentrated.diagnostics["root_deepening_branch_count"] == 2
    assert concentrated.diagnostics["root_adaptive_cycle_limit"] == 3
    assert diffuse.diagnostics["root_consideration_count"] == 9
    assert diffuse.diagnostics["root_deepening_branch_count"] == 8
    assert diffuse.diagnostics["root_adaptive_cycle_limit"] == 2


def test_quantile_deterministic_continuation_is_forced_not_full_depth_table() -> None:
    board = chess.Board("8/8/8/8/8/8/PPP5/K2k4 w - - 0 1")
    result = AdaptiveExpectedValue(
        Style(),
        1.0,
        StaticEvaluator(),
        cdepth=3,
        deepening_quantile=0.01,
    ).search_result(board)

    assert result.diagnostics["root_consideration_count"] > 1
    assert result.diagnostics["root_deepening_branch_count"] == 1
    assert result.diagnostics["root_forced_continuation"] is True
    assert result.diagnostics["root_adaptive_cycle_limit"] is None
    assert result.diagnostics["quantile_forced_nodes"] >= 1
    assert len(result.selected_uci) == 1


def test_objectively_forced_root_follows_sole_legal_move() -> None:
    board = chess.Board("8/8/8/8/8/8/8/K1k5 w - - 0 1")
    result = AdaptiveExpectedValue(
        Style(),
        1.0,
        StaticEvaluator(),
        cdepth=2,
        deepening_quantile=0.8,
    ).search_result(board)

    assert result.diagnostics["root_legal_move_count"] == 1
    assert result.diagnostics["root_forced_continuation"] is True
    assert result.diagnostics["objective_forced_nodes"] >= 1
    assert result.selected_uci == ("a1a2",)
    assert next(move for move in result.moves if move.uci == "a1a2").selected_for_refinement


def test_subjectively_forced_root_follows_sole_considered_move() -> None:
    board = chess.Board()
    result = _ControlledConsiderationSearch(
        Style(),
        1.0,
        StaticEvaluator(),
        cdepth=2,
        deepening_quantile=0.8,
        considered_count=1,
    ).search_result(board)

    considered = [move for move in result.moves if move.in_consideration_set]
    outside = [move for move in result.moves if not move.in_consideration_set]
    assert result.diagnostics["root_legal_move_count"] > 1
    assert result.diagnostics["root_consideration_count"] == 1
    assert result.diagnostics["root_deepening_branch_count"] == 1
    assert result.diagnostics["root_forced_continuation"] is True
    assert result.diagnostics["subjective_forced_nodes"] >= 1
    assert len(considered) == 1
    assert considered[0].selected_for_refinement
    assert all(move.qwa_unavailable_reason == "outside_consideration_set" for move in outside)


def test_cdepth_three_can_use_extra_cycle_in_narrow_positions() -> None:
    board = chess.Board("8/8/8/8/8/8/P7/K1k5 w - - 0 1")
    cdepth_two = AdaptiveExpectedValue(
        Style(),
        1.0,
        StaticEvaluator(),
        cdepth=2,
        deepening_quantile=1.0,
    ).search_result(board)
    cdepth_three = AdaptiveExpectedValue(
        Style(),
        1.0,
        StaticEvaluator(),
        cdepth=3,
        deepening_quantile=1.0,
    ).search_result(board)

    two_cycles = max(move.deepened_cycles for move in cdepth_two.moves if move.in_consideration_set)
    three_cycles = max(move.deepened_cycles for move in cdepth_three.moves if move.in_consideration_set)
    assert cdepth_two.diagnostics["root_adaptive_cycle_limit"] == 2
    assert cdepth_three.diagnostics["root_adaptive_cycle_limit"] == 3
    assert two_cycles <= 2
    assert three_cycles == 3
    assert max(move.recursive_plies_used for move in cdepth_three.moves) > max(
        move.recursive_plies_used for move in cdepth_two.moves
    )
    assert {move.principal_variation for move in cdepth_two.moves} != {
        move.principal_variation for move in cdepth_three.moves
    }


def test_phase_specific_cache_keys_prevent_cross_phase_reuse() -> None:
    board = chess.Board("8/8/8/8/8/8/PPP5/K2k4 w - - 0 1")
    search = _ForcedConsiderationSearch(
        Style(),
        1.0,
        StaticEvaluator(),
        cdepth=3,
        deepening_quantile=1.0,
        forced_counts={"__non_root__": 10},
    )
    key = search.position_key(board)

    search._future_subjective_value(board, 4, 1)  # noqa: SLF001
    search._future_subjective_value(board, 4, 2)  # noqa: SLF001

    assert (key, 4, 1) in search._value_cache  # noqa: SLF001
    assert (key, 4, 0) in search._value_cache  # noqa: SLF001
    assert (key, 3, 1) in search._value_cache  # noqa: SLF001


def test_quantile_selection_is_preserved_when_adaptive_depth_is_positive() -> None:
    board = chess.Board("8/8/8/8/8/8/PPP5/K2k4 w - - 0 1")
    quantile = 0.55
    search = AdaptiveExpectedValue(
        Style(),
        1.0,
        StaticEvaluator(),
        cdepth=2,
        deepening_quantile=quantile,
    )
    landscape = search.landscape(board)
    expected_count = deepening_count_for_quantile(
        (record.probability for record in sorted(landscape.considered_records, key=lambda record: record.probability, reverse=True)),
        quantile,
    )
    result = search.search_result(board)

    assert result.diagnostics["root_adaptive_cycle_limit"] == 2
    assert len(result.selected_uci) == expected_count

def test_search_and_player_transient_caches_can_be_cleared_after_materialization() -> None:
    board = chess.Board("7k/8/8/8/8/8/6K1/7R w - - 0 1")
    evaluator = StaticEvaluator()
    player = ThermoPlayer("white", chess.WHITE, Style(), beta=1.0, cdepth=2)

    analysis = player.analyze(board, evaluator)
    result = analysis.search_result
    serialized_before_clear = result.as_dict()
    search = player.search(evaluator)

    assert player._analysis_cache  # noqa: SLF001
    assert search._result_cache  # noqa: SLF001
    assert search._value_cache  # noqa: SLF001
    assert search._selection_cache  # noqa: SLF001
    assert search._static_cache  # noqa: SLF001
    assert search._context_cache  # noqa: SLF001
    assert search._landscape_cache  # noqa: SLF001

    player.clear_transient_caches()

    assert not player._analysis_cache  # noqa: SLF001
    assert not search._static_cache  # noqa: SLF001
    assert not search._context_cache  # noqa: SLF001
    assert not search._landscape_cache  # noqa: SLF001
    assert not search._response_landscape_cache  # noqa: SLF001
    assert not search._value_cache  # noqa: SLF001
    assert not search._selection_cache  # noqa: SLF001
    assert not search._future_value_cache  # noqa: SLF001
    assert not search._future_selection_cache  # noqa: SLF001
    assert not search._recursive_result_cache  # noqa: SLF001
    assert not search._result_cache  # noqa: SLF001
    assert result.as_dict() == serialized_before_clear

    player.clear_transient_caches()


def test_board_context_cache_reuses_same_position_request() -> None:
    board = chess.Board()
    search = AdaptiveExpectedValue(Style(), 1.0, StaticEvaluator())

    ctx1 = search._get_board_context(board)  # noqa: SLF001
    ctx2 = search._get_board_context(board)  # noqa: SLF001

    assert ctx1 == board_context(board)
    assert ctx1 is ctx2
    assert search.diagnostics.board_context_requests == 2
    assert search.diagnostics.board_context_computations == 1
    assert search.diagnostics.board_context_cache_hits == 1
    assert search.diagnostics.board_context_requests == (
        search.diagnostics.board_context_computations
        + search.diagnostics.board_context_cache_hits
    )


def test_board_context_cache_is_position_based_not_object_identity() -> None:
    first = chess.Board()
    second = chess.Board(first.fen())
    search = AdaptiveExpectedValue(Style(), 1.0, StaticEvaluator())

    ctx1 = search._get_board_context(first)  # noqa: SLF001
    ctx2 = search._get_board_context(second)  # noqa: SLF001

    assert ctx1 is ctx2
    assert search.diagnostics.board_context_computations == 1
    assert search.diagnostics.board_context_cache_hits == 1


def test_board_context_cache_key_distinguishes_state_affecting_context() -> None:
    search = AdaptiveExpectedValue(Style(), 1.0, StaticEvaluator())
    white_to_move = chess.Board()
    black_to_move = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    castling = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    no_castling = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    en_passant = chess.Board()
    en_passant.push_san("e4")
    no_en_passant = chess.Board(en_passant.fen())
    no_en_passant.ep_square = None

    for board in (white_to_move, black_to_move, castling, no_castling, en_passant, no_en_passant):
        search._get_board_context(board)  # noqa: SLF001

    assert search.diagnostics.board_context_computations == 6
    assert search.diagnostics.board_context_cache_hits == 0


def test_board_context_cache_cleanup_leaves_search_reusable() -> None:
    board = chess.Board()
    search = AdaptiveExpectedValue(Style(), 1.0, StaticEvaluator())

    search._get_board_context(board)  # noqa: SLF001
    assert search._context_cache  # noqa: SLF001
    search.clear_transient_caches()

    assert not search._context_cache  # noqa: SLF001
    assert search.diagnostics.board_context_cache_entries == 0
    search._get_board_context(board)  # noqa: SLF001
    assert search._context_cache  # noqa: SLF001


def test_serial_root_branch_contexts_are_cleared_between_candidates() -> None:
    board = chess.Board("7k/8/8/8/8/8/6K1/7R w - - 0 1")
    evaluator = StaticEvaluator()
    search = AdaptiveExpectedValue(Style(), 1.0, evaluator, cdepth=2, search_workers=1)

    result = search.search_result(board)
    lifecycle = search._branch_cache_lifecycle  # noqa: SLF001

    assert len(result.moves) >= 2
    assert len(lifecycle) >= 2
    assert all(sum(snapshot["before_clear"].values()) > 0 for snapshot in lifecycle)
    assert all(sum(snapshot["after_clear"].values()) == 0 for snapshot in lifecycle)
    assert all(snapshot["after_clear"]["context"] == 0 for snapshot in lifecycle)
    assert search._result_cache  # noqa: SLF001
    assert all(
        key[0] == search.position_key(board)
        for key in search._value_cache  # noqa: SLF001
    )


def test_root_candidates_are_materialized_in_one_branch_pass() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    search = _ForcedConsiderationSearch(
        Style(center=1.5, development=1.0),
        2.0,
        evaluator,
        cdepth=2,
        deepening_quantile=0.2,
        search_workers=1,
        forced_counts={"__non_root__": 5},
    )

    result = search.search_result(board)
    considered = [move for move in result.moves if move.in_consideration_set]
    deepened = [move for move in considered if move.selected_for_refinement]

    assert len(considered) > 1
    assert deepened
    assert result.diagnostics["root_candidate_branch_evaluations"] == len(considered)
    assert result.diagnostics["root_candidate_recursive_evaluations"] == len(deepened)
    assert len(search._branch_cache_lifecycle) == len(considered)  # noqa: SLF001


def test_serial_and_parallel_root_materialization_counters_match() -> None:
    board = chess.Board("7k/8/8/8/8/8/6K1/7R w - - 0 1")
    evaluator = StaticEvaluator()

    serial = AdaptiveExpectedValue(Style(), 1.0, evaluator, cdepth=2, search_workers=1).search_result(board)
    parallel = AdaptiveExpectedValue(
        Style(),
        1.0,
        evaluator,
        cdepth=2,
        search_workers=2,
        parallel_min_branches=1,
    ).search_result(board)

    serial_values = {move.uci: move.g_tilde for move in serial.moves if move.in_consideration_set}
    parallel_values = {move.uci: move.g_tilde for move in parallel.moves if move.in_consideration_set}
    assert serial_values.keys() == parallel_values.keys()
    for uci, value in serial_values.items():
        assert parallel_values[uci] == pytest.approx(value)
    assert serial.diagnostics["root_candidate_branch_evaluations"] == parallel.diagnostics["root_candidate_branch_evaluations"]
    assert serial.diagnostics["root_candidate_recursive_evaluations"] == parallel.diagnostics["root_candidate_recursive_evaluations"]


def test_recursive_backup_uses_side_to_move_minimax_over_available_endpoints() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    search = _ForcedConsiderationSearch(
        Style(center=1.5, development=1.0),
        2.0,
        evaluator,
        cdepth=1,
        deepening_quantile=0.2,
        forced_counts={"__non_root__": 10},
    )
    value = search.expected_value(board)
    selection = search.node_selection(board)
    assert selection is not None
    branches = selection.branches
    deepened_mass = sum(branch.probability for branch in branches if branch.was_deepened)

    assert sum(branch.probability for branch in branches) == pytest.approx(1.0)
    assert 0.0 < deepened_mass < 1.0
    assert value == pytest.approx(side_to_move_backup(board.turn, branches))
    assert value != pytest.approx(mass_preserving_expectation(branches))


def test_root_adaptive_g_tilde_does_not_deepen_unselected_moves() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    search = _ForcedConsiderationSearch(
        Style(center=1.5, development=1.0),
        2.0,
        evaluator,
        cdepth=2,
        deepening_quantile=0.05,
        forced_counts={"__non_root__": 5},
    )
    result = search.search_result(board)
    selected = {move.uci for move in result.moves if move.selected_for_refinement}
    unselected = [move for move in result.moves if not move.selected_for_refinement]

    assert selected
    assert unselected
    for move in result.moves:
        after = board.copy(stack=False)
        after.push(move.move)
        shallow = search._future_subjective_value(after, 0, 1)
        if move.uci in selected:
            assert move.depth_used == 3
        else:
            assert move.depth_used == 0
            assert move.deepened_cycles == 0
            assert move.thermo_g_tilde is None
            assert move.q_tilde is None
            assert move.w_tilde is None
            assert move.a_tilde is None
            assert move.terminal_u == pytest.approx(shallow)
            assert move.g_tilde == pytest.approx(shallow - result.U)


def test_internal_adaptive_backup_uses_minimax_not_weighted_sum() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    search = _ForcedConsiderationSearch(
        Style(center=1.5, development=1.0),
        2.0,
        evaluator,
        cdepth=2,
        deepening_quantile=0.2,
        forced_counts={"__non_root__": 5},
    )
    root_result = search.search_result(board)
    selected = next(move for move in root_result.moves if move.selected_for_refinement)
    after = board.copy(stack=False)
    after.push(selected.move)
    selection = search.node_selection(after, 3)

    assert selection is not None
    assert any(branch.was_deepened for branch in selection.branches)
    assert any(not branch.was_deepened for branch in selection.branches)
    value = search._future_subjective_value(after, 3, 1)
    assert value == pytest.approx(side_to_move_backup(after.turn, selection.branches))
    assert value != pytest.approx(mass_preserving_expectation(selection.branches))


def test_qwa_uses_selected_same_turn_endpoint_and_nontrivial_decomposition() -> None:
    board = chess.Board("8/8/8/8/8/8/PPP5/K2k4 w - - 0 1")
    evaluator = StaticEvaluator()
    player = ThermoPlayer(
        "white",
        chess.WHITE,
        Style(center=1.5, development=1.0),
        beta=2.0,
        cdepth=1,
        deepening_quantile=0.2,
    )
    result = player.evaluate_landscape(board, evaluator)
    move = next(
        candidate
        for candidate in result.moves
        if candidate.selected_for_refinement and candidate.qwa_unavailable_reason is None
    )
    endpoint = chess.Board(move.endpoint_fen)

    assert endpoint.turn == board.turn
    assert move.principal_variation
    assert move.deepened_cycles == 1
    assert move.thermo_g_tilde == pytest.approx(move.q_tilde + move.w_tilde + move.a_tilde)
    assert move.thermo_decomposition_error == pytest.approx(0.0)
    assert not (
        move.q_tilde == pytest.approx(0.0)
        and move.w_tilde == pytest.approx(0.0)
        and move.a_tilde == pytest.approx(move.g_tilde)
    )


@pytest.mark.parametrize("cdepth", [1, 2, 3])
def test_selected_principal_variation_endpoint_is_same_turn_for_cycle_depths(cdepth: int) -> None:
    board = chess.Board("8/8/8/8/8/3k4/7P/4K3 w - - 0 1")
    evaluator = StaticEvaluator()
    result = AdaptiveExpectedValue(
        Style(),
        1.0,
        evaluator,
        cdepth=cdepth,
        deepening_quantile=0.2,
    ).search_result(board)
    move = next(candidate for candidate in result.moves if candidate.selected_for_refinement)
    endpoint = chess.Board(move.endpoint_fen)

    assert endpoint.turn == board.turn
    assert move.qwa_unavailable_reason is None
    assert move.thermo_g_tilde == pytest.approx(move.q_tilde + move.w_tilde + move.a_tilde)


def test_partial_cycle_deepening_uses_thermodynamic_checkpoint() -> None:
    board = chess.Board("8/8/8/8/8/8/P7/K1k5 w - - 0 1")
    evaluator = StaticEvaluator()
    result = AdaptiveExpectedValue(
        Style(),
        1.0,
        evaluator,
        cdepth=3,
        deepening_quantile=0.01,
    ).search_result(board)
    move = next(candidate for candidate in result.moves if candidate.selected_for_refinement)

    assert move.requested_cycles == 3
    assert move.deepened_cycles == 1
    assert move.recursive_plies_used == 3
    assert move.search_endpoint_fen != move.thermodynamic_endpoint_fen
    assert chess.Board(move.thermodynamic_endpoint_fen).turn == board.turn
    assert chess.Board(move.search_endpoint_fen).turn != board.turn
    assert move.thermo_g_tilde == pytest.approx(move.q_tilde + move.w_tilde + move.a_tilde)
    assert move.g_tilde != pytest.approx(move.thermo_g_tilde)


def test_full_cycle_deepening_aligns_search_and_thermodynamic_endpoints() -> None:
    board = chess.Board("8/8/8/8/8/8/P7/K1k5 w - - 0 1")
    evaluator = StaticEvaluator()
    result = AdaptiveExpectedValue(
        Style(),
        1.0,
        evaluator,
        cdepth=3,
        deepening_quantile=1.0,
    ).search_result(board)

    assert result.moves
    for move in result.moves:
        assert move.requested_cycles == 3
        assert move.deepened_cycles <= 3
        if move.deepened_cycles == 3:
            assert move.search_endpoint_fen == move.thermodynamic_endpoint_fen
            assert move.g_tilde == pytest.approx(move.thermo_g_tilde)
            assert move.thermo_g_tilde == pytest.approx(move.q_tilde + move.w_tilde + move.a_tilde)


def test_search_and_player_choice_identities() -> None:
    board = chess.Board("8/8/8/8/8/8/P7/K1k5 w - - 0 1")
    evaluator = StaticEvaluator()
    player = ThermoPlayer("white", chess.WHITE, Style(), beta=1.0, cdepth=2)
    result = player.evaluate_landscape(board, evaluator)
    choice = player.choose_from_result(result)
    chosen = next(move for move in result.moves if move.uci == choice.uci)

    assert choice.move in board.legal_moves
    assert choice.value == pytest.approx(chosen.terminal_u)
    assert choice.delta_u == pytest.approx(chosen.g_tilde)
    assert chosen.thermo_g_tilde == pytest.approx(chosen.q_tilde + chosen.w_tilde + chosen.a_tilde)
    assert result.U == pytest.approx(move_distribution(board, player.style, player.beta, evaluator).expected_value)


def test_black_still_selects_minimum_g_tilde() -> None:
    board = chess.Board()
    board.turn = chess.BLACK
    evaluator = StaticEvaluator()
    player = ThermoPlayer("black", chess.BLACK, Style(), beta=1.0, cdepth=1)
    result = player.evaluate_landscape(board, evaluator)
    choice = player.choose_from_result(result)
    selectable = [move for move in result.moves if move.g_tilde is not None]

    assert selectable
    assert choice.uci == min(selectable, key=lambda move: move.g_tilde).uci


@pytest.mark.parametrize("cdepth", [1, 2, 3, 4])
def test_cdepth_validation_accepts_positive_complete_cycles(cdepth: int) -> None:
    assert validate_cdepth(cdepth) == cdepth
    assert requested_recursive_plies_for_cdepth(cdepth) == 2 * cdepth - 1


@pytest.mark.parametrize("cdepth", [-2, -1, 0, 1.5, "3"])
def test_cdepth_validation_rejects_non_cycle_depths(cdepth: object) -> None:
    with pytest.raises(ValueError, match="complete move-response cycles"):
        validate_cdepth(cdepth)  # type: ignore[arg-type]


def test_future_depth_zero_remains_shallow_subjective_landscape() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    search = AdaptiveExpectedValue(Style(center=1.5, development=1.0), 2.0, evaluator, cdepth=1)
    expected = move_distribution(board, search.style, 2.0, evaluator).expected_value
    assert search._future_subjective_value(board, 0, 0) == pytest.approx(expected)
    assert search.expected_value(board, 0) == pytest.approx(expected)


def test_draw_rules_and_captured_material() -> None:
    config = MatchConfig()
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    insufficient = chess.Board("8/8/8/8/8/8/6k1/6K1 w - - 0 1")
    seventyfive = chess.Board("7k/8/8/8/8/8/8/6KQ w - - 150 76")
    captured = captured_material(chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1"))

    assert _terminal_result(stalemate, False, config) == ("1/2-1/2", "stalemate")
    assert _terminal_result(insufficient, False, config) == ("1/2-1/2", "insufficient_material")
    assert _terminal_result(seventyfive, False, config) == ("1/2-1/2", "seventyfive_move_rule")
    assert captured["balance"] == pytest.approx(9.0)


def test_adaptive_configuration_and_strategy_presets() -> None:
    with pytest.raises(ValueError, match="complete move-response cycles"):
        MatchConfig(cdepth=0)
    with pytest.raises(ValueError):
        MatchConfig(deepening_quantile=0.0)
    assert set(STRATEGY_NAMES) == {
        "material_conservative",
        "pressure_aggressive",
        "tactical_attacker",
        "positional_controller",
    }
    styles = [strategy_style(name).as_dict() for name in STRATEGY_NAMES]
    assert len({tuple(style.items()) for style in styles}) == len(STRATEGY_NAMES)
    assert strategy_style("tactical_attacker").king_pressure > strategy_style("positional_controller").king_pressure
    assert strategy_style("positional_controller").center > strategy_style("tactical_attacker").center


def test_default_match_name_uses_strategy_beta_and_cdepth() -> None:
    config = MatchConfig(
        white_strategy="material_conservative",
        black_strategy="positional_controller",
        beta_white=4.0,
        beta_black=6.5,
        cdepth=1,
    )
    assert config.match_name == default_match_name(config)
    assert config.match_name == "material_conservative_b4_vs_positional_controller_b6p5_cdepth1"


def _branch(uci: str, probability: float, value: float, plies: int = 1) -> LandscapeObservation:
    return LandscapeObservation(
        move=chess.Move.from_uci(uci),
        uci=uci,
        probability=probability,
        observable_value=value,
        was_deepened=plies > 0,
        depth_used=plies,
    )


def test_thermodynamic_decomposition_identities() -> None:
    old = (_branch("a2a3", 0.4, 2.0), _branch("b2b3", 0.6, 4.0))
    new = (_branch("a2a3", 0.7, 3.0), _branch("b2b3", 0.3, 1.0))
    result = decompose_transition(old, new)
    assert result.delta_u == pytest.approx(result.delta_q + result.delta_w + result.delta_a)
    assert result.decomposition_error == pytest.approx(0.0)


def test_same_player_transition_and_cycle_payload() -> None:
    evaluator = StaticEvaluator()
    player = ThermoPlayer("white", chess.WHITE, _quiet_style(), beta=1.0, cdepth=1)
    board = chess.Board()
    old_result = player.evaluate_landscape(board, evaluator)
    board.push_san("e4")
    board.push_san("e5")
    new_result = player.evaluate_landscape(board, evaluator)
    record = _thermodynamic_transition_record(player, old_result, 3, new_result, evaluator)

    assert record["decomposition_type"] == "realized_full_shallow_same_player_transition"
    assert record["realized_delta_u"] == pytest.approx(record["realized_delta_q"] + record["realized_delta_w"] + record["realized_delta_a"])
    assert _realized_cycle_payload({"side": "white"}, "black", "d5", "d7d5") is None


def test_simulation_saves_viewer_ready_panels(tmp_path: Path) -> None:
    config = MatchConfig(
        max_plies=2,
        seed=1,
        cdepth=1,
        games_dir=tmp_path / "games",
        match_name="viewer_payload",
    )
    progress_lines: list[str] = []
    result = simulate_match(
        white_style=_quiet_style(material=1.0, center=0.5),
        black_style=_quiet_style(material=1.0, center=0.5),
        config=config,
        progress=progress_lines.append,
    )
    data = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    states = data["viewer_states"]
    first_move = states[0]["white_panel"]["moves"][0]

    assert data["config"]["cdepth"] == 1
    assert Path(result["csv"]).parent == tmp_path / "games" / "csv"
    assert Path(result["json"]).parent == tmp_path / "games" / "json"
    assert Path(result["pgn"]).parent == tmp_path / "games" / "pgn"
    assert set(data["white_style"]) == set(STRATEGY_FEATURES)
    assert len(states) == len(data["plies"]) + 1
    assert result["result"] == data["result"]
    assert result["terminal_reason"] == data["terminal_reason"]
    assert result["plies_played"] == len(data["plies"])
    assert result["full_moves_played"] == 1
    assert progress_lines[0].startswith("Ply 1 | White to move | fullmove 1")
    assert " | selected " in progress_lines[1]
    assert " | G=" in progress_lines[1]
    assert " | cycles " in progress_lines[1]
    assert progress_lines[1].endswith("s")
    assert states[0]["fen"] == chess.STARTING_FEN
    assert first_move["features"]["lambda_sum"] == pytest.approx(1.0)
    assert first_move["g_tilde"] == pytest.approx(first_move["terminal_u"] - states[0]["white_panel"]["expected_value"])


def test_simulation_saves_static_components_for_deepened_responses(tmp_path: Path) -> None:
    config = MatchConfig(
        max_plies=1,
        seed=1,
        cdepth=1,
        deepening_quantile=0.2,
        games_dir=tmp_path / "games",
        match_name="static_components",
    )
    result = simulate_match(
        white_style=_quiet_style(material=1.0, center=0.5),
        black_style=_quiet_style(material=1.0, center=0.5),
        config=config,
    )
    data = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader(Path(result["csv"]).open(newline="", encoding="utf-8")))
    moves = data["plies"][0]["search_result"]["moves"]
    move = next(move for move in moves if move["responses"])
    response = move["responses"][0]

    assert set(move["static_components_after_move"]) == set(COMPONENT_ORDER)
    assert set(response["static_components"]) == set(COMPONENT_ORDER)
    assert response["board_fen"]
    assert "evaluation_weights" in data["plies"][0]["search_result"]
    assert "requested_cycles" in move
    assert "deepened_cycles" in move
    assert "thermo_g_tilde" in move
    assert "search_endpoint_fen" in move
    assert "thermodynamic_endpoint_fen" in move
    assert "requested_cycles" in csv_rows[0]
    assert "deepened_cycles" in csv_rows[0]
    assert "thermo_g_tilde" in csv_rows[0]
    assert data["plies"][0]["search_result"]["diagnostics"]["root_adaptive_cycle_limit"] == 1
    assert move["total_probability_mass"] == pytest.approx(1.0)
    assert (
        move["deepened_probability_mass"] + move["nondeepened_probability_mass"]
        == pytest.approx(1.0)
    )
    assert sum(reply["probability"] for reply in move["responses"]) == pytest.approx(1.0)
    response_values = [reply["response_value"] for reply in move["responses"]]
    assert move["terminal_u"] == pytest.approx(min(response_values))
    assert any(not reply["selected_for_refinement"] for reply in move["responses"])

    output_dir = tmp_path / "analysis" / "covariance"
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_static_covariances.py",
            str(result["json"]),
            "--output",
            str(output_dir),
            "--json-output",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (output_dir / "static_components.csv").exists()
    assert (output_dir / "static_components.json").exists()


def test_weighted_static_covariance_synthetic_example() -> None:
    probabilities = [0.25, 0.75]
    observations = [
        [1.0, 0.0, 2.0, 0.5, -1.0],
        [3.0, 4.0, 2.0, -0.5, 1.0],
    ]
    mean, covariance = weighted_mean_and_covariance(probabilities, observations)

    assert mean == pytest.approx([2.5, 3.0, 2.0, -0.25, 0.5])
    assert covariance[0][1] == pytest.approx(1.5)
    for row in range(len(COMPONENT_ORDER)):
        for col in range(len(COMPONENT_ORDER)):
            assert covariance[row][col] == pytest.approx(covariance[col][row])
        assert covariance[row][row] >= -1e-12


def test_probability_normalization_and_static_variance_identity() -> None:
    responses = [
        {
            "uci": "a7a6",
            "probability": 0.2,
            "conditional_probability": 0.2,
            "static_value": 1.0,
            "static_components": dict(zip(COMPONENT_ORDER, [1.0, 0.0, 0.0, 0.0, 0.0])),
        },
        {
            "uci": "a7a5",
            "probability": 0.3,
            "conditional_probability": 0.3,
            "static_value": 3.0,
            "static_components": dict(zip(COMPONENT_ORDER, [3.0, 0.0, 0.0, 0.0, 0.0])),
        },
    ]
    probabilities, total, normalized = normalize_probabilities(
        responses,
        retained_probability_mass=0.5,
    )
    row, _detail = candidate_covariance_record(
        game="synthetic",
        ply_entry={"ply": 1, "row": {"uci": "a2a4"}},
        move={
            "uci": "a2a4",
            "san": "a4",
            "probability": 0.4,
            "selected_for_refinement": True,
            "retained_probability_mass": 0.5,
            "responses": responses,
        },
        game_weights={name: 1.0 if name == "material" else 0.0 for name in COMPONENT_ORDER},
        selected_actual_move="a2a4",
    )

    assert total == pytest.approx(0.5)
    assert normalized is True
    assert sum(probabilities) == pytest.approx(1.0)
    assert row is not None
    assert row["variance_static_from_covariance"] == pytest.approx(row["variance_static_direct"])
    assert row["variance_consistency_error"] == pytest.approx(0.0)


def test_old_json_without_static_components_is_explicitly_skipped(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "evaluation_weights": {name: 1.0 for name in COMPONENT_ORDER},
                "plies": [
                    {
                        "ply": 1,
                        "row": {"uci": "e2e4"},
                        "search_result": {
                            "moves": [
                                {
                                    "uci": "e2e4",
                                    "san": "e4",
                                    "probability": 1.0,
                                    "selected_for_refinement": True,
                                    "responses": [
                                        {
                                            "uci": "e7e5",
                                            "conditional_probability": 1.0,
                                            "static_value": 0.1,
                                            "features": {"material": 99.0},
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rows, _details, warnings = analyze_game(path)

    assert rows == []
    assert warnings
    assert "static_components unavailable" in warnings[0]


def _synthetic_divergence_inputs() -> tuple[list[dict[str, object]], dict[str, object]]:
    covariance_rows = [
        {
            "game": "synthetic",
            "ply": "7",
            "move_number": "4",
            "side": "white",
            "player": "White",
            "candidate_san": "Ba3",
            "candidate_uci": "c1a3",
            "candidate_probability": "0.25",
            "candidate_was_actual_move": "False",
            "selected_actual_move": "g1f3",
            "number_of_responses": "2",
            "mean_static_evaluation": "-1.5",
            "variance_static_evaluation": "0.4",
        },
        {
            "game": "synthetic",
            "ply": "7",
            "move_number": "4",
            "side": "white",
            "player": "White",
            "candidate_san": "Nf3",
            "candidate_uci": "g1f3",
            "candidate_probability": "0.5",
            "candidate_was_actual_move": "True",
            "selected_actual_move": "g1f3",
            "number_of_responses": "2",
            "mean_static_evaluation": "0.5",
            "variance_static_evaluation": "0.1",
        },
    ]
    game_json = {
        "plies": [
            {
                "ply": 7,
                "row": {"ply": 7, "move_number": 4, "side": "white", "uci": "g1f3"},
                "search_result": {
                    "moves": [
                        {
                            "san": "Ba3",
                            "uci": "c1a3",
                            "probability": 0.25,
                            "selected_for_refinement": True,
                            "actually_deepened": True,
                            "terminal_u": 2.0,
                            "branch_value": 2.0,
                            "g_tilde": 1.2,
                            "q_tilde": 0.2,
                            "w_tilde": 0.3,
                            "a_tilde": 0.7,
                            "retained_probability_mass": 0.6,
                            "retained_response_count": 2,
                        },
                        {
                            "san": "Nf3",
                            "uci": "g1f3",
                            "probability": 0.5,
                            "selected_for_refinement": True,
                            "actually_deepened": True,
                            "terminal_u": 0.25,
                            "branch_value": 0.25,
                            "g_tilde": -0.1,
                            "q_tilde": 0.0,
                            "w_tilde": 0.0,
                            "a_tilde": -0.1,
                            "retained_probability_mass": 0.8,
                            "retained_response_count": 2,
                        },
                    ]
                },
            }
        ]
    }
    return covariance_rows, game_json


def test_deep_search_divergence_matching_and_delta() -> None:
    covariance_rows, game_json = _synthetic_divergence_inputs()
    rows = divergence_rows(covariance_rows, game_json, game="synthetic")

    assert len(rows) == 2
    ba3 = next(row for row in rows if row["candidate_uci"] == "c1a3")
    assert ba3["terminal_minus_mean_static"] == pytest.approx(3.5)
    assert ba3["abs_terminal_minus_mean_static"] == pytest.approx(3.5)
    assert ba3["retained_probability_mass"] == pytest.approx(0.6)
    assert ba3["g_tilde"] == pytest.approx(1.2)


def test_deep_search_divergence_ranking_and_ply_filtering() -> None:
    covariance_rows, game_json = _synthetic_divergence_inputs()
    rows = divergence_rows(covariance_rows, game_json, game="synthetic")
    ranked = rank_by_abs_divergence(rows, 1)
    ply_rows = rows_for_ply(rows, 7)

    assert ranked[0]["candidate_uci"] == "c1a3"
    assert [row["candidate_uci"] for row in ply_rows] == ["c1a3", "g1f3"]
    assert rows_for_ply(rows, 99) == []


def test_deep_search_divergence_missing_fields_and_duplicates() -> None:
    covariance_rows, game_json = _synthetic_divergence_inputs()
    missing = json.loads(json.dumps(game_json))
    del missing["plies"][0]["search_result"]["moves"][0]["terminal_u"]
    with pytest.raises(DivergenceAnalysisError, match="terminal_u"):
        divergence_rows(covariance_rows, missing, game="synthetic")

    with pytest.raises(DivergenceAnalysisError, match="duplicate covariance"):
        divergence_rows(covariance_rows + [dict(covariance_rows[0])], game_json, game="synthetic")

    duplicate_json = json.loads(json.dumps(game_json))
    duplicate_json["plies"][0]["search_result"]["moves"].append(
        dict(duplicate_json["plies"][0]["search_result"]["moves"][0])
    )
    with pytest.raises(DivergenceAnalysisError, match="duplicate deepened JSON"):
        divergence_rows(covariance_rows, duplicate_json, game="synthetic")


def test_deep_search_divergence_cli_with_generated_game(tmp_path: Path) -> None:
    config = MatchConfig(
        max_plies=1,
        seed=2,
        cdepth=1,
        deepening_quantile=0.2,
        games_dir=tmp_path / "games",
        match_name="divergence_game",
    )
    result = simulate_match(
        white_style=_quiet_style(material=1.0, center=0.5),
        black_style=_quiet_style(material=1.0, center=0.5),
        config=config,
    )
    covariance_dir = tmp_path / "analysis" / "covariance"
    divergence_dir = tmp_path / "analysis" / "deep_search_divergence"
    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_static_covariances.py",
            str(result["json"]),
            "--output",
            str(covariance_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_deep_search_divergence.py",
            str(result["json"]),
            "--covariance-dir",
            str(covariance_dir),
            "--output",
            str(divergence_dir),
            "--top",
            "2",
            "--ply",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = divergence_dir / "divergence_game.csv"
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))

    assert output.exists()
    assert rows
    assert "terminal_minus_mean_static" in rows[0]
    assert "abs_terminal_minus_mean_static" in rows[0]
    assert "candidate_uci" in completed.stdout


def test_scale_diagnostic_script_runs_deterministically() -> None:
    command = [
        sys.executable,
        "scripts/diagnose_feature_scales.py",
        "--positions",
        "12",
        "--seed",
        "7",
    ]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert first.stdout == second.stdout
    assert "| material |" in first.stdout
    assert "| king_pressure |" in first.stdout
