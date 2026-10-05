from __future__ import annotations

import csv
import json
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import chess
import pytest

from thermo_chess.evaluation import EvaluationWeights, StaticEvaluator
from thermo_chess.features import (
    DEFAULT_SCALES,
    board_context,
    center_control,
    center_control_raw,
    development_score,
    development_slots,
    game_phase,
    king_freedom,
    king_safety,
    king_safety_raw,
    king_safe_squares,
    legal_mobility,
    material_balance,
    move_features,
    move_feature_breakdown,
    pawn_structure_counts,
    pawn_structure_score,
    unshielded_king_files,
)
from thermo_chess.measure import (
    KAPPA,
    STRATEGY_FEATURES,
    Style,
    beta_from_temperature,
    boltzmann_probabilities,
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
from thermo_chess.search import AdaptiveExpectedValue, LandscapeObservation, SearchResult, adaptive_breadth, mass_preserving_expectation, requested_recursive_plies_for_cdepth, side_to_move_backup, validate_cdepth
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


@pytest.mark.parametrize(
    ("effective_moves", "adaptive_c", "legal_moves", "expected"),
    [(20.0, 0.3, 30, 6), (3.0, 0.3, 20, 1), (20.0, 2.0, 7, 7), (1.0, 0.01, 5, 1)],
)
def test_adaptive_breadth_rule(effective_moves: float, adaptive_c: float, legal_moves: int, expected: int) -> None:
    assert adaptive_breadth(effective_moves, adaptive_c, legal_moves) == expected


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


def test_recursive_backup_uses_side_to_move_minimax_over_available_endpoints() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    search = AdaptiveExpectedValue(
        Style(center=1.5, development=1.0),
        2.0,
        evaluator,
        cdepth=1,
        adaptive_c=0.2,
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
    search = AdaptiveExpectedValue(
        Style(center=1.5, development=1.0),
        2.0,
        evaluator,
        cdepth=2,
        adaptive_c=0.05,
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
    search = AdaptiveExpectedValue(
        Style(center=1.5, development=1.0),
        2.0,
        evaluator,
        cdepth=2,
        adaptive_c=0.2,
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
    board = chess.Board()
    evaluator = StaticEvaluator()
    player = ThermoPlayer(
        "white",
        chess.WHITE,
        Style(center=1.5, development=1.0),
        beta=2.0,
        cdepth=1,
        adaptive_c=0.2,
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
        adaptive_c=0.2,
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
        adaptive_c=0.01,
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
        adaptive_c=10.0,
    ).search_result(board)

    assert result.moves
    for move in result.moves:
        assert move.requested_cycles == 3
        assert move.deepened_cycles == 3
        assert move.search_endpoint_fen == move.thermodynamic_endpoint_fen
        assert move.g_tilde == pytest.approx(move.thermo_g_tilde)
        assert move.thermo_g_tilde == pytest.approx(move.q_tilde + move.w_tilde + move.a_tilde)


def test_search_and_player_choice_identities() -> None:
    board = chess.Board("7k/8/8/8/8/8/6K1/7R w - - 0 1")
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
        MatchConfig(adaptive_c=0.0)
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
    result = simulate_match(
        white_style=_quiet_style(material=1.0, center=0.5),
        black_style=_quiet_style(material=1.0, center=0.5),
        config=config,
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
    assert states[0]["fen"] == chess.STARTING_FEN
    assert first_move["features"]["lambda_sum"] == pytest.approx(1.0)
    assert first_move["g_tilde"] == pytest.approx(first_move["terminal_u"] - states[0]["white_panel"]["expected_value"])


def test_simulation_saves_static_components_for_deepened_responses(tmp_path: Path) -> None:
    config = MatchConfig(
        max_plies=1,
        seed=1,
        cdepth=1,
        adaptive_c=0.2,
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
    move = next(
        move
        for move in data["plies"][0]["search_result"]["moves"]
        if move["responses"]
    )
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
        adaptive_c=0.2,
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
