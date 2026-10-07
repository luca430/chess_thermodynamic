#!/usr/bin/env python3
"""Benchmark entropy-adaptive expected values on a representative position."""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thermo_chess.evaluation import StaticEvaluator
from thermo_chess.features import board_context, legal_mobility
from thermo_chess.measure import Style, move_distribution
from thermo_chess.search import AdaptiveExpectedValue


REPRESENTATIVE_FEN = "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 8"


class CountingEvaluator(StaticEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.component_calls = 0

    def evaluate(self, board: chess.Board, context=None) -> float:
        self.calls += 1
        return super().evaluate(board, context=context)

    def components(self, board: chess.Board, context=None):
        self.component_calls += 1
        return super().components(board, context=context)


def full_expected_value(
    board: chess.Board,
    style: Style,
    beta: float,
    evaluator: CountingEvaluator,
    plies: int,
) -> tuple[float, int]:
    if plies == 0 or board.is_game_over(claim_draw=True):
        return evaluator.evaluate(board), 1
    landscape = move_distribution(board, style, beta, evaluator)
    nodes = 1
    value = 0.0
    for record in landscape.records:
        child = board.copy(stack=False)
        child.push(record.move)
        child_value, child_nodes = full_expected_value(
            child, style, beta, evaluator, plies - 1
        )
        value += record.probability * child_value
        nodes += child_nodes
    return value, nodes


def old_legal_mobility(board: chess.Board, color: chess.Color) -> int:
    probe = board.copy(stack=False)
    probe.turn = color
    probe.ep_square = None
    if probe.is_checkmate() or probe.is_stalemate() or probe.is_insufficient_material():
        return 0
    return probe.legal_moves.count()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", default=REPRESENTATIVE_FEN)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--deepening-quantile", type=float, default=0.8)
    parser.add_argument("--max-cdepth", type=int, default=4)
    parser.add_argument("--context-iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.deepening_quantile <= 0.0 or args.deepening_quantile > 1.0:
        raise SystemExit("--deepening-quantile must satisfy 0 < q <= 1")
    if args.max_cdepth < 1:
        raise SystemExit("--max-cdepth must be at least 1")
    if args.context_iterations < 1:
        raise SystemExit("--context-iterations must be at least 1")

    board = chess.Board(args.fen)
    style = Style()
    mobility_boards = [
        board,
        chess.Board(),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
        chess.Board("8/P7/8/8/8/8/8/K1k5 w - - 0 1"),
    ]
    started = time.perf_counter()
    old_copies = 0
    old_total = 0
    for _ in range(args.context_iterations):
        for mobility_board in mobility_boards:
            for color in (chess.WHITE, chess.BLACK):
                old_total += old_legal_mobility(mobility_board, color)
                old_copies += 1
    old_mobility_elapsed = time.perf_counter() - started
    mobility_stats = {
        "legal_mobility_requests": 0,
        "legal_mobility_direct_counts": 0,
        "legal_mobility_probe_counts": 0,
        "legal_mobility_board_copies": 0,
    }
    started = time.perf_counter()
    new_total = 0
    for _ in range(args.context_iterations):
        for mobility_board in mobility_boards:
            for color in (chess.WHITE, chess.BLACK):
                new_total += legal_mobility(mobility_board, color, stats=mobility_stats)
    new_mobility_elapsed = time.perf_counter() - started
    if old_total != new_total:
        raise SystemExit("optimized legal_mobility differed from old reference")
    print(
        "mobility "
        f"iterations={args.context_iterations * len(mobility_boards) * 2} "
        f"old_seconds={old_mobility_elapsed:.3f} "
        f"new_seconds={new_mobility_elapsed:.3f} "
        f"speedup={old_mobility_elapsed / new_mobility_elapsed:.2f} "
        f"old_board_copies={old_copies} "
        f"new_board_copies={mobility_stats['legal_mobility_board_copies']} "
        f"direct={mobility_stats['legal_mobility_direct_counts']} "
        f"probe={mobility_stats['legal_mobility_probe_counts']}",
        flush=True,
    )
    context_stats = {
        "center_control_raw_computations": 0,
        "king_safety_raw_computations": 0,
        "development_slots_computations": 0,
        "king_safe_squares_computations": 0,
        "legal_mobility_computations": 0,
        "legal_mobility_requests": 0,
        "legal_mobility_direct_counts": 0,
        "legal_mobility_probe_counts": 0,
        "legal_mobility_board_copies": 0,
        "pawn_structure_counts_computations": 0,
    }
    started = time.perf_counter()
    for _ in range(args.context_iterations):
        board_context(board, stats=context_stats)
    elapsed = time.perf_counter() - started
    print(
        "context "
        f"iterations={args.context_iterations} "
        f"seconds={elapsed:.3f} "
        f"contexts_per_second={args.context_iterations / elapsed:.1f} "
        + " ".join(f"{key}={value}" for key, value in context_stats.items()),
        flush=True,
    )
    print("cdepth seconds peak_mb root_candidates    nodes recursive static_req static_miss component_comp combined ctx_req ctx_comp ctx_hits ctx_hit_rate ctx_entries ctx_peak move_dist cache_hits avg_legal avg_neff avg_considered avg_deepened avg_considered_mass avg_deepening_mass   value", flush=True)
    for cdepth in range(1, args.max_cdepth + 1):
        search = AdaptiveExpectedValue(
            style,
            args.beta,
            StaticEvaluator(),
            cdepth=cdepth,
            deepening_quantile=args.deepening_quantile,
        )
        search.begin_diagnostics()
        value = search.expected_value(board)
        diagnostics = search.finish_diagnostics()
        result = search.search_result(board)
        context_requests = int(diagnostics.get("board_context_requests", diagnostics.get("board_context_calls", 0)))
        context_hits = int(diagnostics.get("board_context_cache_hits", 0))
        context_hit_rate = context_hits / context_requests if context_requests else 0.0
        print(
            f"{cdepth:>6}  {diagnostics['elapsed_time']:>7.3f}  "
            f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0:>7.1f}  "
            f"{result.K:>15}  "
            f"{diagnostics['nodes_evaluated']:>7}  "
            f"{diagnostics['recursive_nodes']:>9}  "
            f"{diagnostics.get('static_value_requests', diagnostics['static_evaluations']):>10}  "
            f"{diagnostics['static_evaluations']:>10}  "
            f"{diagnostics.get('static_component_computations', 0):>14}  "
            f"{diagnostics.get('combined_evaluations', 0):>8}  "
            f"{context_requests:>7}  "
            f"{diagnostics.get('board_context_computations', diagnostics.get('board_context_calls', 0)):>8}  "
            f"{context_hits:>8}  "
            f"{context_hit_rate:>12.1%}  "
            f"{diagnostics.get('board_context_cache_entries', 0):>11}  "
            f"{diagnostics.get('board_context_cache_peak_entries', 0):>8}  "
            f"{diagnostics.get('move_distribution_calls', 0):>9}  "
            f"{diagnostics['cache_hits']:>10}  "
            f"{result.diagnostics.get('legal_move_count', 0):>9}  "
            f"{result.N_eff:>8.2f}  "
            f"{result.K:>14}  "
            f"{len(result.selected_uci):>12}  "
            f"{float(result.diagnostics.get('consideration_probability_mass_raw', 0.0)):>19.3f}  "
            f"{float(result.diagnostics.get('deepening_probability_mass_considered', 0.0)):>18.3f}  "
            f"{value:>8.3f}",
            flush=True,
        )

    evaluator = CountingEvaluator()
    started = time.perf_counter()
    value, nodes = full_expected_value(board, style, args.beta, evaluator, plies=2)
    elapsed = time.perf_counter() - started
    print(
        f"full2  {elapsed:>7.3f}  "
        f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0:>7.1f}  "
        f"{'-':>15}  {nodes:>7}  "
        f"{'-':>9}  {evaluator.calls:>10}  {evaluator.calls:>10}  "
        f"{evaluator.component_calls:>14}  {'-':>8}  {'-':>7}  {'-':>8}  "
        f"{'-':>8}  {'-':>12}  {'-':>11}  {'-':>8}  "
        f"{'-':>9}  "
        f"{'-':>10}  {'-':>9}  {'-':>8}  {'-':>14}  {'-':>12}  "
        f"{'-':>19}  {'-':>18}  {value:>8.3f}"
    )


if __name__ == "__main__":
    main()
