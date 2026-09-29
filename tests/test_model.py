from __future__ import annotations

import json
import math
from pathlib import Path

import chess
import pytest

from thermo_chess.evaluation import StaticEvaluator
from thermo_chess.features import board_context, game_phase, move_features, phase_weights, piece_exposure
from thermo_chess.measure import Style, move_distribution, potential, potential_components
from thermo_chess.metrics import effective_number, entropy
from thermo_chess.player import ThermoPlayer
from thermo_chess.search import (
    AdaptiveBranchObservation,
    AdaptiveDepthThresholds,
    AdaptiveExpectedValue,
    SearchResult,
    adaptive_breadth,
    depth_from_effective_moves,
    local_search_depth,
)
from thermo_chess.simulation import (
    STRATEGY_NAMES,
    MatchConfig,
    _terminal_result,
    _termination_reason,
    default_match_name,
    simulate_match,
    strategy_style,
)
from thermo_chess.thermodynamics import adaptive_observable, decompose_transition


def test_move_probabilities_sum_to_one_and_legal_only() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    landscape = move_distribution(board, Style(), 3.0, evaluator)
    assert pytest.approx(sum(record.probability for record in landscape.records)) == 1.0
    legal = {move.uci() for move in board.legal_moves}
    assert {record.uci for record in landscape.records} <= legal


def test_entropy_nonnegative() -> None:
    assert entropy([0.2, 0.3, 0.5]) >= 0.0
    board = chess.Board()
    landscape = move_distribution(board, Style(), 3.0, StaticEvaluator())
    assert landscape.entropy >= 0.0


def test_material_imbalance_changes_sign() -> None:
    evaluator = StaticEvaluator()
    white_up_queen = chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")
    black_up_queen = chess.Board("4kq2/8/8/8/8/8/8/4K3 w - - 0 1")
    assert evaluator.evaluate(white_up_queen) > 0
    assert evaluator.evaluate(black_up_queen) < 0


def test_terminal_positions() -> None:
    evaluator = StaticEvaluator()
    black_mated = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert black_mated.is_checkmate()
    assert evaluator.evaluate(black_mated) > 9000
    assert stalemate.is_stalemate()
    assert evaluator.evaluate(stalemate) == 0.0


def test_changing_lambda_changes_distribution() -> None:
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2")
    evaluator = StaticEvaluator()
    material = move_distribution(board, Style(material=4.0, activity=0.0), 5.0, evaluator)
    activity = move_distribution(board, Style(material=0.0, activity=4.0), 5.0, evaluator)
    probs_a = {record.uci: record.probability for record in material.records}
    probs_b = {record.uci: record.probability for record in activity.records}
    assert any(abs(probs_a[uci] - probs_b[uci]) > 1e-6 for uci in probs_a)


def test_identical_lambda_identical_distribution() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    style = Style(material=2.0, center=1.0)
    first = move_distribution(board, style, 2.5, evaluator)
    second = move_distribution(board, style, 2.5, evaluator)
    assert [record.probability for record in first.records] == pytest.approx(
        [record.probability for record in second.records]
    )


def test_expected_value_is_weighted_average() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    landscape = move_distribution(board, Style(), 3.0, evaluator)
    explicit = sum(record.probability * record.static_after for record in landscape.records)
    assert landscape.expected_value == pytest.approx(explicit)


def test_board_feature_context_preserves_static_and_move_features() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("g1f3")
    after = board.copy(stack=False)
    after.push(move)
    before_context = board_context(board)
    after_context = board_context(after)
    evaluator = StaticEvaluator()

    assert evaluator.evaluate(board, context=before_context) == pytest.approx(
        evaluator.evaluate(board)
    )
    assert move_features(
        board,
        move,
        before_context=before_context,
        after_context=after_context,
        after_board=after,
    ) == pytest.approx(move_features(board, move))


def test_future_depth_zero_is_one_ply_subjective_value() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    style = Style(center=1.5, development=1.0)
    search = AdaptiveExpectedValue(style, 2.0, evaluator, cdepth=1)
    expected = move_distribution(board, style, 2.0, evaluator).expected_value

    assert search._future_subjective_value(board, 0, 0) == pytest.approx(expected)
    assert search.expected_value(board, 0) == pytest.approx(evaluator.evaluate(board))


def test_player_choice_uses_stored_search_result_branch_value() -> None:
    board = chess.Board("7k/8/8/8/8/8/6K1/7R w - - 0 1")
    evaluator = StaticEvaluator()
    player = ThermoPlayer("white", chess.WHITE, Style(), beta=1.0, depth=1)
    result = player.evaluate_landscape(board, evaluator)
    choice = player.choose_from_result(result)
    assert choice.move in board.legal_moves
    assert choice.reply_landscape is None
    chosen = next(move for move in result.moves if move.uci == choice.uci)
    assert choice.value == pytest.approx(chosen.expected_next_U)
    assert choice.delta_u == pytest.approx(chosen.candidate_delta_u)
    assert chosen.candidate_delta_u == pytest.approx(chosen.candidate_delta_q + chosen.candidate_delta_w + chosen.candidate_delta_a)
    assert result.U == pytest.approx(sum(move.probability * move.branch_value for move in result.moves))


def test_adaptive_depth_zero_is_static_evaluation() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    search = AdaptiveExpectedValue(Style(), 1.0, evaluator, depth=0)
    assert search.expected_value(board) == pytest.approx(evaluator.evaluate(board))


def test_adaptive_terminal_position_is_static_at_every_depth() -> None:
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    evaluator = StaticEvaluator()
    expected = evaluator.evaluate(board)
    for depth in range(5):
        search = AdaptiveExpectedValue(Style(), 1.0, evaluator, depth=depth)
        assert search.expected_value(board) == pytest.approx(expected)


def test_effective_number_uniform_and_concentrated() -> None:
    uniform = [0.25] * 4
    assert entropy(uniform) == pytest.approx(math.log(4))
    assert effective_number(entropy(uniform)) == pytest.approx(4.0)
    concentrated = [1.0 - 3e-12, 1e-12, 1e-12, 1e-12]
    assert effective_number(entropy(concentrated)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    ("effective_moves", "adaptive_c", "legal_moves", "expected"),
    [(20.0, 0.3, 30, 6), (3.0, 0.3, 20, 1), (20.0, 2.0, 7, 7), (1.0, 0.01, 5, 1)],
)
def test_adaptive_breadth_rule(
    effective_moves: float, adaptive_c: float, legal_moves: int, expected: int
) -> None:
    assert adaptive_breadth(effective_moves, adaptive_c, legal_moves) == expected




def test_search_result_serialization_preserves_observable() -> None:
    board = chess.Board("7k/8/8/8/8/8/6K1/7R w - - 0 1")
    evaluator = StaticEvaluator()
    search = AdaptiveExpectedValue(Style(), 1.0, evaluator, cdepth=1)
    result = search.search_result(board, player="white", side="white")
    restored = SearchResult.from_dict(result.as_dict())
    assert restored.U == pytest.approx(sum(move.probability * move.branch_value for move in restored.moves))
    assert [move.uci for move in restored.moves] == [move.uci for move in result.moves]


def test_serial_and_parallel_search_results_match() -> None:
    board = chess.Board("7k/8/8/8/8/8/6K1/7R w - - 0 1")
    evaluator = StaticEvaluator()
    serial = AdaptiveExpectedValue(Style(), 1.0, evaluator, cdepth=1, search_workers=1).search_result(board)
    parallel = AdaptiveExpectedValue(Style(), 1.0, evaluator, cdepth=1, search_workers=2, parallel_min_branches=1).search_result(board)
    assert parallel.diagnostics["parallel_branches"] >= len(parallel.moves)
    assert serial.U == pytest.approx(parallel.U)
    for a, b in zip(serial.moves, parallel.moves):
        assert a.uci == b.uci
        assert a.probability == pytest.approx(b.probability)
        assert a.branch_value == pytest.approx(b.branch_value)


def test_search_workers_one_is_serial() -> None:
    result = AdaptiveExpectedValue(Style(), 1.0, StaticEvaluator(), cdepth=1, search_workers=1).search_result(chess.Board("7k/8/8/8/8/8/6K1/7R w - - 0 1"))
    assert result.diagnostics["parallel_branches"] == 0


def test_value_objective_weight_favors_side_to_move_goal() -> None:
    evaluator = StaticEvaluator()
    zero_style = Style(
        material=0.0,
        preservation=0.0,
        king_restriction=0.0,
        king_pressure=0.0,
        check=0.0,
        mate=0.0,
        activity=0.0,
        king_safety=0.0,
        center=0.0,
        promotion=0.0,
    )

    black_board = chess.Board()
    black_board.push_san("e4")
    black_landscape = move_distribution(
        black_board, zero_style, 1.0, evaluator, value_objective_weight=1.0
    )
    black_low = min(black_landscape.records, key=lambda record: record.static_after)
    black_high = max(black_landscape.records, key=lambda record: record.static_after)
    assert black_low.probability > black_high.probability

    white_board = chess.Board()
    white_landscape = move_distribution(
        white_board, zero_style, 1.0, evaluator, value_objective_weight=1.0
    )
    white_low = min(white_landscape.records, key=lambda record: record.static_after)
    white_high = max(white_landscape.records, key=lambda record: record.static_after)
    assert white_high.probability > white_low.probability


def test_terminal_result_reports_standard_chess_draw_rules() -> None:
    config = MatchConfig()
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    insufficient = chess.Board("8/8/8/8/8/8/6k1/6K1 w - - 0 1")
    seventyfive = chess.Board("7k/8/8/8/8/8/8/6KQ w - - 150 76")

    assert _terminal_result(stalemate, False, config) == ("1/2-1/2", "stalemate")
    assert _terminal_result(insufficient, False, config) == (
        "1/2-1/2",
        "insufficient_material",
    )
    assert _terminal_result(seventyfive, False, config) == (
        "1/2-1/2",
        "seventyfive_move_rule",
    )


def test_claimable_draw_position_keeps_search_k_positive_when_not_terminal() -> None:
    board = chess.Board()
    for san in (
        "e4 Nf6 e5 Ne4 Nc3 Nxc3 dxc3 e6 Qd4 Nc6 Qe4 Bc5 "
        "Bd2 Kf8 Bd3 Be7 Nf3 g5 h3 a6 b4 b5 Rb1 h5 "
        "g3 Bb7 a3 Bc8 Ra1 Bb7 Rb1 Bc8 Ra1 Bb7"
    ).split():
        board.push_san(san)

    assert board.legal_moves.count() > 0
    assert board.is_game_over(claim_draw=False) is False
    assert board.is_game_over(claim_draw=True) is True
    assert _termination_reason(
        board,
        MatchConfig(stop_only_on_mate_or_stalemate=True),
    ) is None
    assert _termination_reason(
        board,
        MatchConfig(stop_only_on_mate_or_stalemate=False),
    ) == "threefold_repetition_claim"

    search = AdaptiveExpectedValue(Style(), 1.0, StaticEvaluator(), cdepth=1)
    search.expected_value(board)
    selection = search.node_selection(board)

    assert selection is not None
    assert selection.expanded_count >= 1


def test_cdepth_zero_is_static_evaluation() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    search = AdaptiveExpectedValue(Style(), 1.0, evaluator, cdepth=0)
    assert search.expected_value(board) == pytest.approx(evaluator.evaluate(board))


def test_cdepth_one_accurate_matches_full_move_response_expectation() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    style = Style()
    landscape = move_distribution(board, style, 1.5, evaluator)
    search = AdaptiveExpectedValue(style, 1.5, evaluator, cdepth=1, adaptive_c=0.1)
    explicit = 0.0
    for record in landscape.records:
        after = board.copy(stack=False)
        after.push(record.move)
        reply_landscape = move_distribution(
            after, style, 1.5, evaluator, value_objective_weight=1.0
        )
        reply_expected = sum(
            reply.probability * reply.static_after for reply in reply_landscape.records
        )
        explicit += record.probability * reply_expected
    assert search.expected_value(board) == pytest.approx(explicit)


def test_accurate_mode_preserves_unexpanded_probability_mass() -> None:
    board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1")
    evaluator = StaticEvaluator()
    search = AdaptiveExpectedValue(_quiet_style(), 1.0, evaluator, cdepth=1, adaptive_c=0.1)
    actual = search.expected_value(board)
    landscape = search.landscape(board)
    selection = search.node_selection(board)
    assert selection is not None
    selected = set(selection.selected_uci)
    assert 0 < len(selected) < len(landscape.records)

    explicit = 0.0
    fallback_mass = 0.0
    for record in landscape.records:
        after = board.copy(stack=False)
        after.push(record.move)
        reply_landscape = move_distribution(
            after, _quiet_style(), 1.0, evaluator, value_objective_weight=1.0
        )
        branch_value = sum(reply.probability * reply.static_after for reply in reply_landscape.records)
        if record.uci not in selected:
            fallback_mass += record.probability
        explicit += record.probability * branch_value

    assert fallback_mass > 0.0
    assert actual == pytest.approx(explicit)




def test_top_k_uses_immediate_static_evaluation_order() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    style = Style(center=3.0, development=1.0)
    search = AdaptiveExpectedValue(style, 2.0, evaluator, cdepth=1, adaptive_c=0.2)
    search.expected_value(board)
    landscape = search.landscape(board)
    selection = search.node_selection(board)
    assert selection is not None
    expected_k = adaptive_breadth(landscape.effective_moves, 0.2, len(landscape.records))
    expected = tuple(
        record.uci
        for record in sorted(landscape.records, key=lambda item: item.static_after, reverse=True)[:expected_k]
    )
    assert selection.selected_uci == expected


def test_response_k_is_independent_and_static_evaluation_ordered() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    style = Style(center=3.0, development=1.0)
    search = AdaptiveExpectedValue(style, 2.0, evaluator, cdepth=1, adaptive_c=0.2)
    search.expected_value(board)
    selection = search.node_selection(board)
    assert selection is not None
    selected_branch = next(branch for branch in selection.branches if branch.was_deepened)
    after = board.copy(stack=False)
    after.push(selected_branch.move)
    reply_landscape = search.response_landscape(after)
    expected_k = adaptive_breadth(reply_landscape.effective_moves, 0.2, len(reply_landscape.records))
    expected = {
        record.uci
        for record in sorted(reply_landscape.records, key=lambda item: item.static_after)[:expected_k]
    }
    observed = {
        reply.uci
        for reply in selected_branch.response_branches
        if reply.selected_for_refinement
    }
    assert selected_branch.response_k == expected_k
    assert observed == expected


def test_accurate_mode_does_not_renormalize_or_drop_omitted_replies() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    style = Style(center=3.0, development=1.0)
    search = AdaptiveExpectedValue(style, 2.0, evaluator, cdepth=1, adaptive_c=0.05)
    search.expected_value(board)
    selection = search.node_selection(board)
    assert selection is not None
    branch = next(branch for branch in selection.branches if branch.was_deepened)
    assert branch.response_k is not None
    assert branch.response_k < len(branch.response_branches)
    assert sum(reply.probability for reply in branch.response_branches) == pytest.approx(1.0)
    assert any(not reply.selected_for_refinement for reply in branch.response_branches)


def test_candidate_thermodynamics_averages_all_replies_after_consuming_cycle() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    style = Style(center=3.0, development=1.0)
    search = AdaptiveExpectedValue(
        style,
        2.0,
        evaluator,
        cdepth=1,
        adaptive_c=0.05,
        candidate_thermo_mode="all",
    )
    result = search.search_result(board)

    candidate = next(
        move for move in result.moves if move.selected_for_refinement and move.responses
    )
    after = board.copy(stack=False)
    after.push(candidate.move)
    reply_landscape = search.response_landscape(after)
    expected_k = adaptive_breadth(
        reply_landscape.effective_moves,
        0.05,
        len(reply_landscape.records),
    )
    ranked = sorted(
        reply_landscape.records,
        key=lambda reply: reply.static_after,
        reverse=after.turn == chess.WHITE,
    )[:expected_k]
    selected = {reply.uci for reply in ranked}

    expected_next_u = 0.0
    for reply in reply_landscape.records:
        next_board = after.copy(stack=False)
        next_board.push(reply.move)
        if reply.uci in selected and not next_board.is_game_over(claim_draw=True):
            future_u = move_distribution(next_board, style, 2.0, evaluator).expected_value
        else:
            future_u = reply.static_after
        expected_next_u += reply.probability * future_u

    assert candidate.response_entropy is None
    assert candidate.response_N_eff == pytest.approx(reply_landscape.effective_moves)
    assert candidate.response_K == expected_k
    assert len(candidate.responses) == len(reply_landscape.records)
    assert [response.uci for response in candidate.responses] == [
        reply.uci for reply in reply_landscape.records
    ]
    assert sum(response.probability for response in candidate.responses) == pytest.approx(1.0)
    assert [response.probability for response in candidate.responses] == pytest.approx(
        [reply.probability for reply in reply_landscape.records]
    )
    assert {
        response.uci for response in candidate.responses if response.selected_for_refinement
    } == selected
    assert all(response.depth_used == 0 for response in candidate.responses)
    assert candidate.expected_next_U == pytest.approx(expected_next_u)
    assert candidate.candidate_delta_u == pytest.approx(candidate.expected_next_U - result.U)
    assert candidate.candidate_delta_u == pytest.approx(
        candidate.candidate_delta_q + candidate.candidate_delta_w + candidate.candidate_delta_a
    )


def test_selected_candidate_thermo_mode_only_enriches_chosen_move() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    player = ThermoPlayer(
        "white",
        chess.WHITE,
        Style(center=3.0, development=1.0),
        beta=2.0,
        cdepth=1,
        adaptive_c=0.05,
        candidate_thermo_mode="selected",
    )
    choice = player.choose(board, evaluator)

    chosen = next(
        move for move in choice.search_result.moves if move.uci == choice.uci
    )
    nonchosen = [
        move for move in choice.search_result.moves if move.uci != choice.uci
    ]

    assert chosen.candidate_delta_q is not None
    assert chosen.candidate_delta_u == pytest.approx(
        chosen.candidate_delta_q + chosen.candidate_delta_w + chosen.candidate_delta_a
    )
    assert all(move.candidate_delta_q is None for move in nonchosen)


def test_candidate_thermo_mode_does_not_change_move_decision() -> None:
    board = chess.Board()
    board.push_san("e4")
    evaluator = StaticEvaluator()
    choices = []

    for mode in ("selected", "refined", "all"):
        player = ThermoPlayer(
            "black",
            chess.BLACK,
            strategy_style("positional_controller"),
            beta=6.0,
            cdepth=1,
            adaptive_c=0.3,
            candidate_thermo_mode=mode,
        )
        choices.append(player.choose(board, evaluator).uci)

    assert choices == ["g8f6", "g8f6", "g8f6"]


def test_refined_candidate_thermo_mode_enriches_delta_u_top_k_moves() -> None:
    board = chess.Board(
        "r3k2r/p1pb1p1p/3pp3/2b1P2p/7P/2PB1NP1/P1PB1P2/1R2K2R b Kkq - 3 16"
    )
    evaluator = StaticEvaluator()
    search = AdaptiveExpectedValue(
        strategy_style("positional_controller"),
        6.0,
        evaluator,
        cdepth=1,
        adaptive_c=0.3,
    )
    result = search.search_result(board)
    selected = set(result.selected_uci)
    ranked_by_delta = sorted(
        result.moves,
        key=lambda move: move.candidate_delta_u,
        reverse=board.turn == chess.WHITE,
    )
    expected_detailed = {move.uci for move in ranked_by_delta[: result.K]}

    assert result.search_mode == "accurate"
    assert result.K == len(selected)
    assert selected
    assert expected_detailed != selected
    for move in result.moves:
        if move.uci in expected_detailed:
            assert move.candidate_delta_q is not None
            assert move.candidate_delta_u == pytest.approx(
                move.candidate_delta_q + move.candidate_delta_w + move.candidate_delta_a
            )
        else:
            assert move.candidate_delta_q is None
            assert move.responses == ()


def test_cdepth_two_recurses_only_after_complete_cycle() -> None:
    board = chess.Board()
    evaluator = StaticEvaluator()
    search = AdaptiveExpectedValue(Style(center=3.0), 2.0, evaluator, cdepth=2, adaptive_c=0.05)
    search.expected_value(board)
    selection = search.node_selection(board)
    assert selection is not None
    branch = next(branch for branch in selection.branches if branch.was_deepened)
    assert branch.depth_used == 2
    assert any(reply.depth_used == 1 for reply in branch.response_branches)
    assert all(reply.depth_used in (0, 1) for reply in branch.response_branches)


def test_depths_two_through_four_are_probability_weighted() -> None:
    board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1")
    evaluator = StaticEvaluator()
    values = [
        AdaptiveExpectedValue(_quiet_style(), 1.0, evaluator, depth=depth, adaptive_c=0.1).expected_value(board)
        for depth in (2, 3, 4)
    ]
    assert all(math.isfinite(value) for value in values)


def test_white_maximizes_and_black_minimizes_adaptive_candidate_values() -> None:
    evaluator = StaticEvaluator()
    white_board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1")
    white = ThermoPlayer("white", chess.WHITE, _quiet_style(), depth=1)
    white_choice = white.choose(white_board, evaluator)
    assert white_choice.value == max(
        candidate.value for candidate in white_choice.analysis.candidates
    )

    black_board = chess.Board("4k3/4r3/8/8/8/8/8/4K3 b - - 0 1")
    black = ThermoPlayer("black", chess.BLACK, _quiet_style(), depth=1)
    black_choice = black.choose(black_board, evaluator)
    assert black_choice.value == min(
        candidate.value for candidate in black_choice.analysis.candidates
    )


def test_adaptive_configuration_validation() -> None:
    with pytest.raises(ValueError):
        MatchConfig(depth=-1)
    with pytest.raises(ValueError):
        MatchConfig(adaptive_c=0.0)
    with pytest.raises(ValueError):
        MatchConfig(white_solidness=-0.1)
    with pytest.raises(ValueError):
        MatchConfig(black_solidness=1.1)
    assert MatchConfig().cdepth == 1
    assert MatchConfig().depth == 1
    assert MatchConfig().search_mode == "accurate"
    assert MatchConfig().adaptive_c == pytest.approx(0.3)
    assert MatchConfig().candidate_thermo_mode == "refined"


def test_builtin_strategy_presets_are_complete_and_distinct() -> None:
    assert set(STRATEGY_NAMES) == {
        "material_conservative",
        "activity_aggressive",
        "tactical_attacker",
        "positional_controller",
    }
    styles = [strategy_style(name).as_dict() for name in STRATEGY_NAMES]
    assert len({tuple(style.items()) for style in styles}) == len(STRATEGY_NAMES)
    assert strategy_style("tactical_attacker").check > strategy_style(
        "positional_controller"
    ).check
    assert strategy_style("positional_controller").center > strategy_style(
        "tactical_attacker"
    ).center


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError):
        MatchConfig(white_strategy="unknown")
    with pytest.raises(ValueError):
        strategy_style("unknown")


@pytest.mark.parametrize(
    ("effective_moves", "expected_depth"),
    [
        (1.0, 4),
        (4.0, 4),
        (4.0001, 3),
        (8.0, 3),
        (8.0001, 2),
        (15.0, 2),
        (15.0001, 1),
        (30.0, 1),
    ],
)
def test_effective_moves_select_local_depth(
    effective_moves: float, expected_depth: int
) -> None:
    assert depth_from_effective_moves(effective_moves) == expected_depth


def test_local_depth_never_exceeds_configured_maximum() -> None:
    for effective_moves in (1.0, 4.0, 6.0, 12.0, 20.0):
        assert local_search_depth(2, effective_moves) <= 2


def test_child_remaining_depth_can_only_decrease() -> None:
    thresholds = AdaptiveDepthThresholds(4.0, 8.0, 15.0)
    for remaining_depth in range(1, 5):
        for effective_moves in (2.0, 6.0, 12.0, 20.0):
            selected_depth = local_search_depth(
                remaining_depth, effective_moves, thresholds
            )
            child_remaining_depth = selected_depth - 1
            assert child_remaining_depth < remaining_depth
            assert child_remaining_depth >= 0


def test_depth_threshold_configuration_must_be_increasing() -> None:
    with pytest.raises(ValueError):
        MatchConfig(depth4_max_neff=8.0, depth3_max_neff=4.0)


def _quiet_style(**overrides: float) -> Style:
    values = {
        "material": 0.0,
        "preservation": 0.0,
        "king_restriction": 0.0,
        "king_pressure": 0.0,
        "check": 0.0,
        "mate": 0.0,
        "activity": 0.0,
        "king_safety": 0.0,
        "center": 0.0,
        "promotion": 0.0,
    }
    values.update(overrides)
    return Style(**values)


def test_preservation_is_the_only_exchange_safety_parameter() -> None:
    board = chess.Board("4k3/8/8/8/3p4/4N3/8/4K3 w - - 0 1")
    assert "safety" not in Style().as_dict()
    assert "safety" not in move_features(board, chess.Move.from_uci("e3g4"))


def test_material_capture_feature_uses_mover_perspective_material_delta() -> None:
    board = chess.Board("4k3/8/4n3/3P4/8/8/8/4K3 w - - 0 1")
    features = move_features(board, chess.Move.from_uci("d5e6"))
    assert features["material"] == pytest.approx(3.0)


def test_moving_hanging_knight_to_safety_preserves_material() -> None:
    board = chess.Board("4k3/8/8/8/3p4/4N3/8/4K3 w - - 0 1")
    features = move_features(board, chess.Move.from_uci("e3g4"))
    assert features["preservation"] == pytest.approx(3.0)


def test_moving_safe_queen_onto_pawn_attack_is_negative_preservation() -> None:
    board = chess.Board("4k3/8/8/8/3p4/8/4Q3/4K3 w - - 0 1")
    features = move_features(board, chess.Move.from_uci("e2e3"))
    assert features["preservation"] == pytest.approx(-9.0)


def test_pawn_attacks_defended_knight_still_has_positive_exchange_exposure() -> None:
    board = chess.Board("4k3/8/8/8/3p4/4N3/3B4/4K3 b - - 0 1")
    assert piece_exposure(board, chess.E3, chess.WHITE) == pytest.approx(2.0)


def test_bishop_for_knight_with_recapture_is_exchange_neutral() -> None:
    board = chess.Board("4k3/8/8/8/2b5/4N3/3B4/4K3 b - - 0 1")
    assert piece_exposure(board, chess.E3, chess.WHITE) == pytest.approx(0.0)


def test_pawn_attacks_undefended_queen_has_large_exposure() -> None:
    board = chess.Board("4k3/8/8/8/3p4/4Q3/8/4K3 b - - 0 1")
    assert piece_exposure(board, chess.E3, chess.WHITE) == pytest.approx(9.0)


def test_king_restriction_feature_rewards_reducing_enemy_king_moves() -> None:
    board = chess.Board("7k/8/5K2/8/8/8/6Q1/8 w - - 0 1")
    features = move_features(board, chess.Move.from_uci("g2g6"))
    assert features["king_restriction"] > 0.0


def test_check_and_mate_features_are_explicit() -> None:
    check_board = chess.Board("4k3/8/8/8/3p4/8/4Q3/4K3 w - - 0 1")
    check_features = move_features(check_board, chess.Move.from_uci("e2e3"))
    assert check_features["check"] == 1.0
    assert check_features["mate"] == 0.0

    mate_board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 w - - 0 1")
    mate_features = move_features(mate_board, chess.Move.from_uci("g7f8"))
    assert mate_features["check"] == 1.0
    assert mate_features["mate"] == 1.0


def test_move_features_are_perspective_consistent_for_white_and_black() -> None:
    white_board = chess.Board("4k3/8/4n3/3P4/8/8/8/4K3 w - - 0 1")
    black_board = chess.Board("4k3/8/8/8/3p4/4N3/8/4K3 b - - 0 1")
    assert move_features(white_board, chess.Move.from_uci("d5e6"))["material"] == pytest.approx(3.0)
    assert move_features(black_board, chess.Move.from_uci("d4e3"))["material"] == pytest.approx(3.0)


def test_unmoved_king_move_loses_both_castling_rights() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    features = move_features(board, chess.Move.from_uci("e1f1"))
    assert features["castle_preserve"] == -2.0


def test_original_rook_move_loses_corresponding_castling_right() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    features = move_features(board, chess.Move.from_uci("h1h2"))
    assert features["castle_preserve"] == -1.0


def test_castling_has_no_preservation_penalty_and_sets_indicator() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    features = move_features(board, chess.Move.from_uci("e1g1"))
    assert features["castle_preserve"] == 0.0
    assert features["castle"] == 1.0


def test_capturing_original_rook_denies_castling_right() -> None:
    board = chess.Board("4k2r/8/8/8/8/8/7R/4K3 w k - 0 1")
    features = move_features(board, chess.Move.from_uci("h2h8"))
    assert features["castle_deny"] == 1.0


def test_castling_features_use_mover_perspective_for_both_colors() -> None:
    white = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    black = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    assert move_features(white, chess.Move.from_uci("h1h2"))["castle_preserve"] == -1.0
    assert move_features(black, chess.Move.from_uci("h8h7"))["castle_preserve"] == -1.0


def _move_record(board: chess.Board, style: Style, uci: str) -> dict[str, object]:
    landscape = move_distribution(board, style, beta=1.0, evaluator=StaticEvaluator())
    for record in landscape.records:
        if record.uci == uci:
            return record.as_dict()
    raise AssertionError(f"move {uci} not found")


def test_early_king_move_has_large_castling_preservation_contribution() -> None:
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK1NR w KQkq - 0 1")
    record = _move_record(board, Style(castle_preserve=2.0), "e1f1")
    features = record["features"]
    assert features["castle_preserve_raw"] == -2.0
    assert features["castle_preserve_phase_weight"] > 0.9
    assert features["castle_preserve_effective_weight"] > 1.8
    assert features["castle_preserve_contribution"] < -3.6


def test_original_rook_move_has_castling_preservation_contribution() -> None:
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR w KQkq - 0 1")
    record = _move_record(board, Style(castle_preserve=2.0), "h1h2")
    features = record["features"]
    assert features["castle_preserve_raw"] == -1.0
    assert features["castle_preserve_contribution"] == pytest.approx(-2.0)


def test_castling_has_no_castling_preservation_contribution() -> None:
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    record = _move_record(board, Style(castle_preserve=2.0), "e1g1")
    features = record["features"]
    assert features["castle_preserve_raw"] == 0.0
    assert features["castle_preserve_contribution"] == pytest.approx(0.0)


def test_castling_preservation_fades_with_game_phase() -> None:
    style = Style(castle_preserve=2.0, solidness=0.5)
    early_weights = phase_weights(0.1)
    late_weights = phase_weights(0.9)
    early = {
        "castle_preserve": -1.0,
        "game_phase": 0.1,
        "castle_phase_weight": early_weights[1],
    }
    late = {
        "castle_preserve": -1.0,
        "game_phase": 0.9,
        "castle_phase_weight": late_weights[1],
    }
    early_record = _move_record(
        chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR w KQkq - 0 1"),
        Style(castle_preserve=2.0),
        "h1h2",
    )
    assert early_record["features"]["castle_preserve_phase_weight"] == pytest.approx(
        1.0 - early_record["features"]["game_phase"]
    )
    early_contribution = potential_components(style, early)[0]
    late_contribution = potential_components(style, late)[0]
    assert early_contribution < late_contribution
    assert abs(early_contribution) > abs(late_contribution)


def test_solidness_strengthens_early_castling_preservation() -> None:
    weights = phase_weights(0.25)
    features = {
        "castle_preserve": -1.0,
        "game_phase": 0.25,
        "castle_phase_weight": weights[1],
    }
    flexible = Style(castle_preserve=2.0, solidness=0.0)
    solid = Style(castle_preserve=2.0, solidness=1.0)
    assert potential_components(solid, features)[0] < potential_components(flexible, features)[0]


def test_all_builtin_strategies_preserve_castling_rights_meaningfully() -> None:
    for name in STRATEGY_NAMES:
        assert strategy_style(name).castle_preserve >= 1.0


def test_tactical_gain_can_outweigh_castling_preservation_penalty() -> None:
    style = Style(castle_preserve=3.0, mate=80.0)
    tactical_king_move = {
        "castle_preserve": -2.0,
        "game_phase": 0.0,
        "castle_phase_weight": 0.0,
        "mate": 1.0,
    }
    quiet_move = {
        "castle_preserve": 0.0,
        "game_phase": 0.0,
        "castle_phase_weight": 0.0,
        "mate": 0.0,
    }
    assert potential(style, tactical_king_move) > potential(style, quiet_move)


def test_game_phase_is_bounded_and_increases_with_development() -> None:
    start = chess.Board()
    developed = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "d3", "Bc5"):
        developed.push_san(san)
    assert game_phase(start) == pytest.approx(0.0)
    assert 0.0 <= game_phase(start) <= 1.0
    assert 0.0 <= game_phase(developed) <= 1.0
    assert game_phase(developed) > game_phase(start)


def test_phase_weight_progression() -> None:
    early = phase_weights(0.1)
    middle = phase_weights(0.5)
    late = phase_weights(0.9)
    assert early[0] > middle[0] > late[0]
    assert middle[1] > early[1]
    assert middle[1] > late[1]
    assert early[2] < middle[2] < late[2]


def test_solidness_validation_and_zero_reproduces_base_potential() -> None:
    with pytest.raises(ValueError):
        Style(solidness=-0.01)
    with pytest.raises(ValueError):
        Style(solidness=1.01)
    board = chess.Board()
    features = move_features(board, chess.Move.from_uci("g1f3"))
    style = Style(development=2.0, phase_castle=2.0, phase_attack=2.0, solidness=0.0)
    base, phase, total = potential_components(style, features)
    assert phase != 0.0
    assert total == pytest.approx(base)
    assert potential(style, features) == pytest.approx(base)


def test_solidness_emphasizes_development_early_and_attack_late() -> None:
    solid = _quiet_style(development=1.0, phase_attack=1.0, solidness=1.0)
    ordinary = _quiet_style(development=1.0, phase_attack=1.0, solidness=0.0)
    template = {
        "game_phase": 0.1,
        "development_phase_weight": 0.9,
        "castle_phase_weight": 0.36,
        "attack_phase_weight": 0.1,
        "phase_castle_feature": 0.0,
    }
    early_development = {
        **template, "development_feature": 1.0, "phase_attack_feature": 0.0
    }
    early_attack = {
        **template, "development_feature": 0.0, "phase_attack_feature": 1.0
    }
    late_development = {
        **early_development,
        "game_phase": 0.9,
        "development_phase_weight": 0.1,
        "attack_phase_weight": 0.9,
    }
    late_attack = {
        **early_attack,
        "game_phase": 0.9,
        "development_phase_weight": 0.1,
        "attack_phase_weight": 0.9,
    }
    assert potential(solid, early_development) > potential(solid, early_attack)
    assert potential(solid, late_attack) > potential(solid, late_development)
    assert potential(ordinary, early_development) == potential(ordinary, early_attack)
    assert potential(ordinary, late_attack) == potential(ordinary, late_development)


def test_preservation_lambda_increases_probability_of_saving_hanging_material() -> None:
    board = chess.Board("4k3/8/8/8/3p4/4N3/8/4K3 w - - 0 1")
    evaluator = StaticEvaluator()
    neutral = move_distribution(board, _quiet_style(), 1.0, evaluator)
    preserving = move_distribution(board, _quiet_style(preservation=4.0), 1.0, evaluator)
    neutral_probs = {record.uci: record.probability for record in neutral.records}
    preserving_probs = {record.uci: record.probability for record in preserving.records}
    assert preserving_probs["e3g4"] > neutral_probs["e3g4"]


def test_king_restriction_lambda_increases_probability_of_restricting_move() -> None:
    board = chess.Board("7k/8/5K2/8/8/8/6Q1/8 w - - 0 1")
    evaluator = StaticEvaluator()
    neutral = move_distribution(board, _quiet_style(), 1.0, evaluator)
    restricting = move_distribution(board, _quiet_style(king_restriction=3.0), 1.0, evaluator)
    neutral_probs = {record.uci: record.probability for record in neutral.records}
    restricting_probs = {record.uci: record.probability for record in restricting.records}
    assert restricting_probs["g2g6"] > neutral_probs["g2g6"]


def _branch(
    uci: str, probability: float, value: float, depth: int = 1
) -> AdaptiveBranchObservation:
    return AdaptiveBranchObservation(
        move=chess.Move.from_uci(uci),
        uci=uci,
        probability=probability,
        adaptive_branch_value=value,
        was_deepened=depth > 0,
        depth_used=depth,
    )


def test_thermodynamic_midpoint_decomposition_is_exact() -> None:
    old = (_branch("a2a3", 0.4, 2.0), _branch("b2b3", 0.6, 4.0))
    new = (_branch("a2a3", 0.7, 3.0), _branch("b2b3", 0.3, 1.0))
    result = decompose_transition(old, new)
    expected_q = 0.5 * (2.0 + 3.0) * (0.7 - 0.4) + 0.5 * (
        4.0 + 1.0
    ) * (0.3 - 0.6)
    expected_w = 0.5 * (0.4 + 0.7) * (3.0 - 2.0) + 0.5 * (
        0.6 + 0.3
    ) * (1.0 - 4.0)
    assert result.delta_q == pytest.approx(expected_q)
    assert result.delta_w == pytest.approx(expected_w)
    assert result.delta_a == pytest.approx(0.0)
    assert result.delta_u == pytest.approx(
        result.delta_q + result.delta_w + result.delta_a
    )
    assert result.decomposition_error == pytest.approx(0.0)


def test_empty_common_support_is_entirely_accessibility() -> None:
    old = (_branch("a2a3", 1.0, 2.0),)
    new = (_branch("b7b6", 1.0, 5.0),)
    result = decompose_transition(old, new)
    assert result.common_support_size == 0
    assert result.delta_q == 0.0
    assert result.delta_w == 0.0
    assert result.delta_a == pytest.approx(result.delta_u)


def test_move_appearance_and_disappearance_enter_accessibility() -> None:
    old = (_branch("a2a3", 0.5, 2.0), _branch("b2b3", 0.5, 4.0))
    new = (_branch("a2a3", 0.5, 2.0), _branch("c2c3", 0.5, 8.0))
    result = decompose_transition(old, new)
    assert result.delta_q == pytest.approx(0.0)
    assert result.delta_w == pytest.approx(0.0)
    assert result.delta_a == pytest.approx(0.5 * 8.0 - 0.5 * 4.0)
    assert result.old_only_probability_mass == pytest.approx(0.5)
    assert result.new_only_probability_mass == pytest.approx(0.5)


def test_adaptive_branch_values_are_exact_terms_of_adaptive_u() -> None:
    board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1")
    search = AdaptiveExpectedValue(_quiet_style(), 1.0, StaticEvaluator(), depth=3)
    value = search.expected_value(board)
    branches = search.branch_observations(board)
    assert branches
    assert adaptive_observable(branches) == pytest.approx(value)
    selection = search.node_selection(board)
    assert selection is not None
    assert branches == selection.branches
    assert all(branch.depth_used > 0 for branch in branches if branch.was_deepened)
    assert all(branch.depth_used == 0 for branch in branches if not branch.was_deepened)


def test_adaptive_depth_changes_are_allowed_to_enter_work() -> None:
    board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1")
    evaluator = StaticEvaluator()
    shallow = AdaptiveExpectedValue(_quiet_style(), 1.0, evaluator, depth=1)
    deep = AdaptiveExpectedValue(_quiet_style(), 1.0, evaluator, depth=3)
    shallow_u = shallow.expected_value(board)
    deep_u = deep.expected_value(board)
    result = decompose_transition(
        shallow.branch_observations(board),
        deep.branch_observations(board),
        shallow_u,
        deep_u,
    )
    assert result.common_support_size == board.legal_moves.count()
    assert result.delta_a == pytest.approx(0.0)
    assert result.delta_w == pytest.approx(result.delta_u)


def test_terminal_static_fallback_is_boundary_accessibility() -> None:
    terminal = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    evaluator = StaticEvaluator()
    search = AdaptiveExpectedValue(Style(), 1.0, evaluator, depth=4)
    terminal_u = search.expected_value(terminal)
    result = decompose_transition((), search.branch_observations(terminal), 0.0, terminal_u)
    assert result.delta_q == 0.0
    assert result.delta_w == 0.0
    assert result.delta_a == pytest.approx(terminal_u)
    assert result.new_support_size == 1
    assert result.boundary_accessibility == pytest.approx(0.0)
    assert result.decomposition_error == pytest.approx(0.0)


def test_default_match_name_reflects_strategy_beta_solidness_and_cdepth() -> None:
    config = MatchConfig(
        white_strategy="material_conservative",
        black_strategy="positional_controller",
        beta_white=4.0,
        beta_black=6.5,
        white_solidness=0.2,
        black_solidness=None,
        cdepth=3,
    )

    assert config.match_name == default_match_name(config)
    assert config.match_name == (
        "material_conservative_b4_s0p2_vs_"
        "positional_controller_b6p5_s0p8_cdepth3"
    )
    assert MatchConfig(match_name="custom_name").match_name == "custom_name"


def test_simulation_saves_viewer_ready_panels(tmp_path: Path) -> None:
    config = MatchConfig(
        max_plies=2,
        seed=1,
        depth=1,
        results_dir=tmp_path / "results",
        games_dir=tmp_path / "games",
        match_name="viewer_payload",
        white_solidness=0.2,
        black_solidness=0.9,
    )
    result = simulate_match(
        white_style=_quiet_style(material=1.0, center=0.5),
        black_style=_quiet_style(material=1.0, center=0.5),
        config=config,
    )
    data = json.loads(Path(result["json"]).read_text(encoding="utf-8"))

    states = data["viewer_states"]
    assert data["config"]["cdepth"] == 1
    assert data["config"]["depth"] == 1
    assert data["config"]["search_mode"] == "accurate"
    assert data["config"]["adaptive_c"] == pytest.approx(0.3)
    assert data["white_style"]["solidness"] == pytest.approx(0.2)
    assert data["black_style"]["solidness"] == pytest.approx(0.9)
    assert len(states) == len(data["plies"]) + 1
    assert states[0]["fen"] == chess.STARTING_FEN
    assert states[0]["order"] == [move["uci"] for move in states[0]["white_panel"]["moves"]]

    first_move = states[0]["white_panel"]["moves"][0]
    assert {"san", "uci", "probability", "static_after", "delta_u"} <= first_move.keys()
    assert first_move["delta_u"] == pytest.approx(first_move["delta_q"] + first_move["delta_w"] + first_move["delta_a"])
    assert first_move["thermo_decomposition_error"] == pytest.approx(0.0, abs=1e-9)
    assert "reply_landscape" not in first_move
    assert states[0]["actual_search_result"]["side"] == "white"
    assert states[0]["black_panel"]["counterfactual_available"] is False
    assert states[0]["black_panel"]["moves"] == []
    assert states[1]["actual_search_result"]["side"] == "black"
    assert states[1]["white_panel"]["counterfactual_available"] is False
    assert states[0]["white_panel"]["search_depth"] == 1
    assert states[0]["white_panel"]["selected_depth"] == 1
    assert states[0]["white_panel"]["expanded_count"] >= 1
    assert "nodes_evaluated" in states[0]["white_panel"]["diagnostics"]
    assert "selected_for_deeper_analysis" in first_move

    assert data["plies"][0]["search_result"] == states[0]["actual_search_result"]

    first_row = data["plies"][0]["row"]
    assert first_row["U_after_move"] - first_row["U_current"] == pytest.approx(
        first_row["delta_u"]
    )
    assert first_row["search_depth"] == 1
    assert first_row["N_eff"] == pytest.approx(
        first_row["effective_moves_current"]
    )
    assert first_row["selected_depth"] == 1
    assert first_row["K_expanded"] >= 1
    assert first_row["delta_U"] is None
    assert first_row["delta_Q"] is None
    assert first_row["delta_W"] is None
    assert first_row["delta_A"] is None
    assert first_row["N_eff_before"] == pytest.approx(first_row["N_eff"])

    transitions = data["thermodynamic_transitions"]
    assert transitions == []

    assert sum(
        move["probability"] * move["branch_value"]
        for move in data["plies"][0]["search_result"]["moves"]
    ) == pytest.approx(first_row["U_current"])

    white_panel_deltas = [move["delta_u"] for move in states[0]["white_panel"]["moves"]]
    assert white_panel_deltas == sorted(white_panel_deltas, reverse=True)
    assert states[0]["white_panel"]["sort_direction"] == "descending"

    assert states[0]["black_panel"]["moves"] == []
    assert states[0]["black_panel"]["sort_direction"] == "none"

    assert states[1]["white_panel"]["moves"] == []
    assert states[1]["white_panel"]["sort_direction"] == "none"

    black_panel_after_black_to_move = [
        move["delta_u"] for move in states[1]["black_panel"]["moves"]
    ]
    assert black_panel_after_black_to_move == sorted(black_panel_after_black_to_move)
    assert states[1]["black_panel"]["sort_direction"] == "ascending"
