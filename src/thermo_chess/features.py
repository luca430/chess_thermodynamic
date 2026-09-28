"""Move and board features for the toy thermodynamic chess model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import chess

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]
EXTENDED_CENTER = [
    chess.C3,
    chess.D3,
    chess.E3,
    chess.F3,
    chess.C4,
    chess.F4,
    chess.C5,
    chess.F5,
    chess.C6,
    chess.D6,
    chess.E6,
    chess.F6,
]

INITIAL_MINOR_SQUARES = {
    chess.WHITE: (chess.B1, chess.C1, chess.F1, chess.G1),
    chess.BLACK: (chess.B8, chess.C8, chess.F8, chess.G8),
}
INITIAL_CENTRAL_PAWN_SQUARES = {
    chess.WHITE: (chess.D2, chess.E2),
    chess.BLACK: (chess.D7, chess.E7),
}
INITIAL_MAJOR_SQUARES = {
    chess.WHITE: (chess.A1, chess.D1, chess.H1),
    chess.BLACK: (chess.A8, chess.D8, chess.H8),
}


def material(board: chess.Board, color: chess.Color) -> float:
    return sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in PIECE_VALUES.items()
    )


def material_balance(board: chess.Board) -> float:
    """White material minus black material."""

    return material(board, chess.WHITE) - material(board, chess.BLACK)


def material_for_side(board: chess.Board, side: chess.Color) -> float:
    """Own material minus opponent material from `side`'s perspective."""

    return material(board, side) - material(board, not side)


def legal_mobility(board: chess.Board, color: chess.Color) -> int:
    """Count legal moves for `color` without mutating the caller's board."""

    probe = board.copy(stack=False)
    probe.turn = color
    if probe.is_checkmate() or probe.is_stalemate() or probe.is_insufficient_material():
        return 0
    return probe.legal_moves.count()


def attacked_center_score(board: chess.Board, color: chess.Color) -> float:
    score = 0.0
    for square in CENTER:
        if board.is_attacked_by(color, square):
            score += 1.0
    for square in EXTENDED_CENTER:
        if board.is_attacked_by(color, square):
            score += 0.35
    return score


def king_safety(board: chess.Board, color: chess.Color) -> float:
    """Small static king-safety feature from the side's own perspective."""

    king_square = board.king(color)
    if king_square is None:
        return -1.0

    enemy = not color
    attackers = len(board.attackers(enemy, king_square))
    defenders = len(board.attackers(color, king_square))
    adjacent_pressure = 0
    for square in chess.SquareSet(chess.BB_KING_ATTACKS[king_square]):
        if board.is_attacked_by(enemy, square):
            adjacent_pressure += 1

    return (defenders - 1.5 * attackers - 0.25 * adjacent_pressure) / 8.0


def _piece_value_on(board: chess.Board, square: chess.Square) -> float:
    piece = board.piece_at(square)
    if piece is None:
        return 0.0
    return PIECE_VALUES[piece.piece_type]


def _legal_captures_to(
    board: chess.Board, square: chess.Square, color: chess.Color
) -> List[chess.Move]:
    probe = board.copy(stack=False)
    probe.turn = color
    captures = []
    for move in probe.legal_moves:
        if move.to_square == square and probe.is_capture(move):
            captures.append(move)
    return captures


def _exchange_gain_from(board: chess.Board, square: chess.Square, color: chess.Color) -> float:
    return _exchange_gain_from_fen(board.fen(), square, bool(color))


@lru_cache(maxsize=200_000)
def _exchange_gain_from_fen(fen: str, square: chess.Square, color: bool) -> float:
    """Approximate best local material gain from captures onto one square.

    This is a shallow Static Exchange Evaluation-style feature, not a board
    search. It recursively considers only legal captures to `square`, values the
    piece currently on that square, and lets each side decline unprofitable
    continuation captures.
    """

    board = chess.Board(fen)
    side = chess.WHITE if color else chess.BLACK
    best_gain = 0.0
    for move in _legal_captures_to(board, square, side):
        captured_value = _piece_value_on(board, square)
        if captured_value <= 0.0:
            continue
        after = board.copy(stack=False)
        after.turn = side
        after.push(move)
        gain = captured_value - _exchange_gain_from(after, square, not side)
        best_gain = max(best_gain, gain)
    return max(0.0, best_gain)


def exchange_gain(board: chess.Board, square: chess.Square, attacker_side: chess.Color) -> float:
    """Material the attacker can profitably win by opening on `square`.

    The square must contain an opposing non-king piece. Returned values are in
    pawn units: a pawn winning a defended knight is about 2, while a bishop
    trading for a defended knight is about 0.
    """

    piece = board.piece_at(square)
    if piece is None or piece.color == attacker_side or piece.piece_type == chess.KING:
        return 0.0
    return _exchange_gain_from(board, square, attacker_side)


def piece_exposure(board: chess.Board, square: chess.Square, defending_side: chess.Color) -> float:
    """Profitable material available to the opponent on this piece's square."""

    piece = board.piece_at(square)
    if piece is None or piece.color != defending_side or piece.piece_type == chess.KING:
        return 0.0
    return exchange_gain(board, square, not defending_side)


def material_exposure(board: chess.Board, side: chess.Color) -> float:
    return _material_exposure_from_fen(board.fen(), bool(side))


@lru_cache(maxsize=100_000)
def _material_exposure_from_fen(fen: str, side: bool) -> float:
    """Total profitable material currently exposed for `side`.

    This aggregates local exchange losses over side-owned non-king pieces. It
    intentionally measures exchange value, not raw defender counts.
    """

    board = chess.Board(fen)
    color = chess.WHITE if side else chess.BLACK
    return sum(
        piece_exposure(board, square, color)
        for square, piece in board.piece_map().items()
        if piece.color == color and piece.piece_type != chess.KING
    )


def legal_king_moves(board: chess.Board, color: chess.Color) -> int:
    """Count legal king moves for `color` on the current board."""

    king_square = board.king(color)
    if king_square is None:
        return 0
    probe = board.copy(stack=False)
    probe.turn = color
    return sum(1 for move in probe.legal_moves if move.from_square == king_square)


def king_pressure(board: chess.Board, attacking_side: chess.Color) -> float:
    """Squares around the opponent king controlled by `attacking_side`."""

    king_square = board.king(not attacking_side)
    if king_square is None:
        return 0.0
    return float(
        sum(
            1
            for square in chess.SquareSet(chess.BB_KING_ATTACKS[king_square])
            if board.is_attacked_by(attacking_side, square)
        )
    )


def castling_rights_count(board: chess.Board, color: chess.Color) -> int:
    """Number of kingside/queenside castling rights retained by `color`."""

    return int(board.has_kingside_castling_rights(color)) + int(
        board.has_queenside_castling_rights(color)
    )


def _vacated_fraction(
    board: chess.Board,
    color: chess.Color,
    squares: tuple[chess.Square, ...],
    expected_types: frozenset[chess.PieceType],
) -> float:
    undeveloped = sum(
        1
        for square in squares
        if (piece := board.piece_at(square)) is not None
        and piece.color == color
        and piece.piece_type in expected_types
    )
    return 1.0 - undeveloped / len(squares)


def side_development(board: chess.Board, color: chess.Color) -> float:
    """Cheap development score for one side, clamped to [0, 1]."""

    minor = _vacated_fraction(
        board,
        color,
        INITIAL_MINOR_SQUARES[color],
        frozenset((chess.KNIGHT, chess.BISHOP)),
    )
    central_pawns = _vacated_fraction(
        board,
        color,
        INITIAL_CENTRAL_PAWN_SQUARES[color],
        frozenset((chess.PAWN,)),
    )
    major = _vacated_fraction(
        board,
        color,
        INITIAL_MAJOR_SQUARES[color],
        frozenset((chess.ROOK, chess.QUEEN)),
    )
    return min(1.0, max(0.0, 0.60 * minor + 0.25 * central_pawns + 0.15 * major))


def game_phase(board: chess.Board) -> float:
    """Development phase averaged across both colors, in [0, 1]."""

    phase = 0.5 * (
        side_development(board, chess.WHITE)
        + side_development(board, chess.BLACK)
    )
    return min(1.0, max(0.0, phase))


def phase_weights(phase: float) -> tuple[float, float, float]:
    """Return development, consolidation, and attack phase weights."""

    g = min(1.0, max(0.0, phase))
    return 1.0 - g, 4.0 * g * (1.0 - g), g


@dataclass(frozen=True)
class MoveFeatureConfig:
    mobility_scale: float = 30.0
    center_scale: float = 6.0
    king_safety_scale: float = 1.0


def move_features(
    board: chess.Board,
    move: chess.Move,
    config: MoveFeatureConfig | None = None,
) -> Dict[str, float]:
    """Features from the perspective of the side actually making `move`.

    Material and preservation are in pawn units. Activity, center, and
    king-safety retain their small normalized scales; check/mate are indicators;
    king restriction and pressure are counts of king-neighborhood squares.
    """

    config = config or MoveFeatureConfig()
    color = board.turn
    enemy = not color
    before_material = material_for_side(board, color)
    before_exposure = material_exposure(board, color)
    before_enemy_king_moves = legal_king_moves(board, enemy)
    before_king_pressure = king_pressure(board, color)
    before_mobility = legal_mobility(board, color)
    before_enemy_mobility = legal_mobility(board, enemy)
    before_center = attacked_center_score(board, color)
    before_king = king_safety(board, color)
    before_castling = castling_rights_count(board, color)
    before_enemy_castling = castling_rights_count(board, enemy)
    before_development = side_development(board, color)
    phase = game_phase(board)
    development_weight, castle_weight, attack_weight = phase_weights(phase)
    is_castling = board.is_castling(move)

    promotion = 0.0
    if move.promotion:
        promotion = PIECE_VALUES[move.promotion] - PIECE_VALUES[chess.PAWN]

    after = board.copy(stack=False)
    after.push(move)
    material_delta = material_for_side(after, color) - before_material
    preservation = before_exposure - material_exposure(after, color)
    king_restriction = before_enemy_king_moves - legal_king_moves(after, enemy)
    pressure = king_pressure(after, color) - before_king_pressure

    after_mobility = legal_mobility(after, color)
    after_enemy_mobility = legal_mobility(after, enemy)
    activity = (
        (after_mobility - before_mobility)
        - 0.5 * (after_enemy_mobility - before_enemy_mobility)
    ) / config.mobility_scale

    center = (attacked_center_score(after, color) - before_center) / config.center_scale
    king = (king_safety(after, color) - before_king) / max(config.king_safety_scale, 1e-12)
    castle_preserve = float(castling_rights_count(after, color) - before_castling)
    if is_castling:
        castle_preserve = 0.0
    castle_deny = float(
        before_enemy_castling - castling_rights_count(after, enemy)
    )
    castle = 1.0 if is_castling else 0.0
    development_feature = (
        side_development(after, color) - before_development + max(0.0, activity)
    )
    phase_castle_feature = castle + castle_preserve + king
    phase_attack_feature = activity + pressure + float(king_restriction) + (
        1.0 if after.is_check() else 0.0
    )

    return {
        "material": material_delta,
        "preservation": preservation,
        "activity": activity,
        "king_safety": king,
        "king_restriction": float(king_restriction),
        "king_pressure": pressure,
        "center": center,
        "check": 1.0 if after.is_check() else 0.0,
        "mate": 1.0 if after.is_checkmate() else 0.0,
        "promotion": promotion / 8.0,
        "castle_preserve": castle_preserve,
        "castle_deny": castle_deny,
        "castle": castle,
        "game_phase": phase,
        "development_phase_weight": development_weight,
        "castle_phase_weight": castle_weight,
        "attack_phase_weight": attack_weight,
        "development_feature": development_feature,
        "phase_castle_feature": phase_castle_feature,
        "phase_attack_feature": phase_attack_feature,
    }


def white_minus_black_features(board: chess.Board) -> Dict[str, float]:
    return {
        "material": material_balance(board),
        "mobility": (legal_mobility(board, chess.WHITE) - legal_mobility(board, chess.BLACK))
        / 30.0,
        "king_safety": king_safety(board, chess.WHITE) - king_safety(board, chess.BLACK),
        "center": (
            attacked_center_score(board, chess.WHITE)
            - attacked_center_score(board, chess.BLACK)
        )
        / 6.0,
    }
