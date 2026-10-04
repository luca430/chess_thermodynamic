"""Static board descriptors and immediate move features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import chess

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

CENTER = (chess.D4, chess.E4, chess.D5, chess.E5)
EXTENDED_CENTER = (
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
)

INITIAL_MINOR_SLOTS = {
    chess.WHITE: {
        chess.B1: chess.KNIGHT,
        chess.C1: chess.BISHOP,
        chess.F1: chess.BISHOP,
        chess.G1: chess.KNIGHT,
    },
    chess.BLACK: {
        chess.B8: chess.KNIGHT,
        chess.C8: chess.BISHOP,
        chess.F8: chess.BISHOP,
        chess.G8: chess.KNIGHT,
    },
}

PHASE_MAX_NON_PAWN_MATERIAL = 62.0


@dataclass(frozen=True)
class FeatureScales:
    """Pawn-unit conversion factors for all non-material descriptors."""

    passed_pawn: float = 0.30
    connected_pawn: float = 0.10
    isolated_pawn: float = 0.15
    doubled_pawn: float = 0.15
    mobility_per_legal_move: float = 0.03
    center_control: float = 0.10
    extended_center_weight: float = 0.35
    king_safety: float = 0.12
    king_shield_raw: float = 1.0
    king_enemy_zone_control_raw: float = 5.0 / 6.0
    king_unshielded_file_raw: float = 1.25
    development_per_minor: float = 0.15
    castling_move: float = 0.40
    castling_right: float = 0.20
    king_freedom: float = 0.10


DEFAULT_SCALES = FeatureScales()


@dataclass(frozen=True)
class PawnStructureCounts:
    isolated: int
    doubled: int
    passed: int
    connected: int


@dataclass(frozen=True)
class BoardFeatureContext:
    material_by_side: Dict[chess.Color, float]
    material_balance: float
    pawn_structure_by_side: Dict[chess.Color, float]
    pawn_counts_by_side: Dict[chess.Color, PawnStructureCounts]
    mobility_by_side: Dict[chess.Color, int]
    mobility_score_by_side: Dict[chess.Color, float]
    center_by_side: Dict[chess.Color, float]
    center_raw_by_side: Dict[chess.Color, float]
    king_safety_by_side: Dict[chess.Color, float]
    king_safety_raw_by_side: Dict[chess.Color, float]
    castling_rights_by_side: Dict[chess.Color, int]
    development_by_side: Dict[chess.Color, float]
    development_slots_by_side: Dict[chess.Color, int]
    king_safe_squares_by_side: Dict[chess.Color, int]
    king_freedom_by_side: Dict[chess.Color, float]
    game_phase: float

    def material_for_side(self, side: chess.Color) -> float:
        return self.material_by_side[side] - self.material_by_side[not side]


def material_value(board: chess.Board, color: chess.Color) -> float:
    return sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in PIECE_VALUES.items()
    )


def material(board: chess.Board, color: chess.Color) -> float:
    return material_value(board, color)


def material_balance(board: chess.Board) -> float:
    return material_value(board, chess.WHITE) - material_value(board, chess.BLACK)


def material_for_side(board: chess.Board, side: chess.Color) -> float:
    return material_value(board, side) - material_value(board, not side)


def _pawn_rank_direction(color: chess.Color) -> int:
    return 1 if color == chess.WHITE else -1


def pawn_structure_counts(board: chess.Board, color: chess.Color) -> PawnStructureCounts:
    pawns = tuple(board.pieces(chess.PAWN, color))
    by_file = {file_index: [] for file_index in range(8)}
    for square in pawns:
        by_file[chess.square_file(square)].append(chess.square_rank(square))

    isolated = 0
    passed = 0
    connected = 0
    direction = _pawn_rank_direction(color)
    enemy_pawns = tuple(board.pieces(chess.PAWN, not color))

    for square in pawns:
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        adjacent_files = [
            candidate
            for candidate in (file_index - 1, file_index + 1)
            if 0 <= candidate <= 7
        ]
        if not any(by_file[file_] for file_ in adjacent_files):
            isolated += 1
        if any(
            abs(chess.square_rank(other) - rank_index) <= 1
            for file_ in adjacent_files
            for other in board.pieces(chess.PAWN, color)
            if chess.square_file(other) == file_
        ):
            connected += 1

        blocked = False
        for enemy_square in enemy_pawns:
            enemy_file = chess.square_file(enemy_square)
            enemy_rank = chess.square_rank(enemy_square)
            if abs(enemy_file - file_index) <= 1 and (enemy_rank - rank_index) * direction > 0:
                blocked = True
                break
        if not blocked:
            passed += 1

    doubled = sum(max(0, len(ranks) - 1) for ranks in by_file.values())
    return PawnStructureCounts(
        isolated=isolated,
        doubled=doubled,
        passed=passed,
        connected=connected,
    )


def pawn_structure_score(
    board: chess.Board,
    color: chess.Color,
    scales: FeatureScales = DEFAULT_SCALES,
) -> float:
    counts = pawn_structure_counts(board, color)
    return (
        scales.passed_pawn * counts.passed
        + scales.connected_pawn * counts.connected
        - scales.isolated_pawn * counts.isolated
        - scales.doubled_pawn * counts.doubled
    )


def legal_mobility(board: chess.Board, color: chess.Color) -> int:
    probe = board.copy(stack=False)
    probe.turn = color
    probe.ep_square = None
    if (
        probe.is_checkmate()
        or probe.is_stalemate()
        or probe.is_insufficient_material()
    ):
        return 0
    return probe.legal_moves.count()


def mobility_score(
    board: chess.Board,
    color: chess.Color,
    scales: FeatureScales = DEFAULT_SCALES,
) -> float:
    return scales.mobility_per_legal_move * legal_mobility(board, color)


def center_control_raw(
    board: chess.Board,
    color: chess.Color,
    scales: FeatureScales = DEFAULT_SCALES,
) -> float:
    return (
        sum(1 for square in CENTER if board.is_attacked_by(color, square))
        + scales.extended_center_weight
        * sum(1 for square in EXTENDED_CENTER if board.is_attacked_by(color, square))
    )


def center_control(
    board: chess.Board,
    color: chess.Color,
    scales: FeatureScales = DEFAULT_SCALES,
) -> float:
    return scales.center_control * center_control_raw(board, color, scales)


def _king_zone(board: chess.Board, color: chess.Color) -> tuple[chess.Square, ...]:
    king_square = board.king(color)
    if king_square is None:
        return ()
    return tuple(chess.SquareSet(chess.BB_KING_ATTACKS[king_square]))


def unshielded_king_files(board: chess.Board, color: chess.Color) -> int:
    king_square = board.king(color)
    if king_square is None:
        return 0
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    direction = _pawn_rank_direction(color)
    open_files = 0
    for file_index in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        has_forward_pawn = False
        rank = king_rank + direction
        while 0 <= rank <= 7:
            piece = board.piece_at(chess.square(file_index, rank))
            if (
                piece is not None
                and piece.color == color
                and piece.piece_type == chess.PAWN
            ):
                has_forward_pawn = True
                break
            rank += direction
        if not has_forward_pawn:
            open_files += 1
    return open_files


def king_safety(
    board: chess.Board,
    color: chess.Color,
    scales: FeatureScales = DEFAULT_SCALES,
) -> float:
    return scales.king_safety * king_safety_raw(board, color, scales)


def king_safety_raw(
    board: chess.Board,
    color: chess.Color,
    scales: FeatureScales = DEFAULT_SCALES,
) -> float:
    zone = _king_zone(board, color)
    shield = sum(
        1
        for square in zone
        if (piece := board.piece_at(square)) is not None
        and piece.color == color
        and piece.piece_type == chess.PAWN
    )
    enemy_control = sum(1 for square in zone if board.is_attacked_by(not color, square))
    return (
        scales.king_shield_raw * shield
        - scales.king_enemy_zone_control_raw * enemy_control
        - scales.king_unshielded_file_raw * unshielded_king_files(board, color)
    )


def castling_rights_count(board: chess.Board, color: chess.Color) -> int:
    return int(board.has_kingside_castling_rights(color)) + int(
        board.has_queenside_castling_rights(color)
    )


def development_slots(board: chess.Board, color: chess.Color) -> int:
    developed = 0
    for square, expected_type in INITIAL_MINOR_SLOTS[color].items():
        piece = board.piece_at(square)
        if piece is None or piece.color != color or piece.piece_type != expected_type:
            developed += 1
    return developed


def development_score(
    board: chess.Board,
    color: chess.Color,
    scales: FeatureScales = DEFAULT_SCALES,
) -> float:
    return scales.development_per_minor * development_slots(board, color)


def king_safe_squares(
    board: chess.Board,
    king_color: chess.Color,
    attacking_side: chess.Color,
) -> float:
    safe = 0
    for square in _king_zone(board, king_color):
        piece = board.piece_at(square)
        if piece is not None and piece.color == king_color:
            continue
        if board.is_attacked_by(attacking_side, square):
            continue
        safe += 1
    return safe


def king_freedom(
    board: chess.Board,
    king_color: chess.Color,
    attacking_side: chess.Color,
    scales: FeatureScales = DEFAULT_SCALES,
) -> float:
    return scales.king_freedom * king_safe_squares(board, king_color, attacking_side)


def non_pawn_phase_material(board: chess.Board) -> float:
    return sum(
        len(board.pieces(piece_type, color)) * PIECE_VALUES[piece_type]
        for color in (chess.WHITE, chess.BLACK)
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )


def game_phase(board: chess.Board) -> float:
    phase = 1.0 - non_pawn_phase_material(board) / PHASE_MAX_NON_PAWN_MATERIAL
    return min(1.0, max(0.0, phase))


def phase_factors(phase: float) -> Dict[str, float]:
    g = min(1.0, max(0.0, phase))
    return {
        "material": 1.0,
        "center": 1.0 - g,
        "development": 1.0 - g,
        "castling": 1.0 - g,
        "king_safety": 1.0 - g,
        "king_pressure": 1.0,
    }


def phase_weights(phase: float) -> tuple[float, float, float]:
    """Compatibility helper returning development, castling, pressure factors."""

    factors = phase_factors(phase)
    return factors["development"], factors["castling"], factors["king_pressure"]


def board_context(
    board: chess.Board,
    scales: FeatureScales = DEFAULT_SCALES,
) -> BoardFeatureContext:
    material_by_side = {
        chess.WHITE: material_value(board, chess.WHITE),
        chess.BLACK: material_value(board, chess.BLACK),
    }
    pawn_counts_by_side = {
        chess.WHITE: pawn_structure_counts(board, chess.WHITE),
        chess.BLACK: pawn_structure_counts(board, chess.BLACK),
    }
    mobility_by_side = {
        chess.WHITE: legal_mobility(board, chess.WHITE),
        chess.BLACK: legal_mobility(board, chess.BLACK),
    }
    return BoardFeatureContext(
        material_by_side=material_by_side,
        material_balance=material_by_side[chess.WHITE] - material_by_side[chess.BLACK],
        pawn_structure_by_side={
            color: (
                scales.passed_pawn * counts.passed
                + scales.connected_pawn * counts.connected
                - scales.isolated_pawn * counts.isolated
                - scales.doubled_pawn * counts.doubled
            )
            for color, counts in pawn_counts_by_side.items()
        },
        pawn_counts_by_side=pawn_counts_by_side,
        mobility_by_side=mobility_by_side,
        mobility_score_by_side={
            color: scales.mobility_per_legal_move * count
            for color, count in mobility_by_side.items()
        },
        center_by_side={
            chess.WHITE: center_control(board, chess.WHITE, scales),
            chess.BLACK: center_control(board, chess.BLACK, scales),
        },
        center_raw_by_side={
            chess.WHITE: center_control_raw(board, chess.WHITE, scales),
            chess.BLACK: center_control_raw(board, chess.BLACK, scales),
        },
        king_safety_by_side={
            chess.WHITE: king_safety(board, chess.WHITE, scales),
            chess.BLACK: king_safety(board, chess.BLACK, scales),
        },
        king_safety_raw_by_side={
            chess.WHITE: king_safety_raw(board, chess.WHITE, scales),
            chess.BLACK: king_safety_raw(board, chess.BLACK, scales),
        },
        castling_rights_by_side={
            chess.WHITE: castling_rights_count(board, chess.WHITE),
            chess.BLACK: castling_rights_count(board, chess.BLACK),
        },
        development_by_side={
            chess.WHITE: development_score(board, chess.WHITE, scales),
            chess.BLACK: development_score(board, chess.BLACK, scales),
        },
        development_slots_by_side={
            chess.WHITE: development_slots(board, chess.WHITE),
            chess.BLACK: development_slots(board, chess.BLACK),
        },
        king_safe_squares_by_side={
            chess.WHITE: king_safe_squares(board, chess.WHITE, chess.BLACK),
            chess.BLACK: king_safe_squares(board, chess.BLACK, chess.WHITE),
        },
        king_freedom_by_side={
            chess.WHITE: king_freedom(board, chess.WHITE, chess.BLACK, scales),
            chess.BLACK: king_freedom(board, chess.BLACK, chess.WHITE, scales),
        },
        game_phase=game_phase(board),
    )


def castling_transition(
    board: chess.Board,
    move: chess.Move,
    before: BoardFeatureContext,
    after: BoardFeatureContext,
    scales: FeatureScales = DEFAULT_SCALES,
) -> float:
    color = board.turn
    if board.is_castling(move):
        return scales.castling_move
    rights_lost = max(
        0,
        before.castling_rights_by_side[color]
        - after.castling_rights_by_side[color],
    )
    return -scales.castling_right * rights_lost


def castling_transition_raw(
    board: chess.Board,
    move: chess.Move,
    before: BoardFeatureContext,
    after: BoardFeatureContext,
) -> tuple[float, float]:
    color = board.turn
    if board.is_castling(move):
        return 1.0, 0.0
    rights_lost = max(
        0,
        before.castling_rights_by_side[color]
        - after.castling_rights_by_side[color],
    )
    return 0.0, -float(rights_lost)


def move_feature_breakdown(
    board: chess.Board,
    move: chess.Move,
    before_context: BoardFeatureContext | None = None,
    after_context: BoardFeatureContext | None = None,
    after_board: chess.Board | None = None,
    scales: FeatureScales = DEFAULT_SCALES,
) -> Dict[str, Dict[str, float] | float]:
    """Raw descriptors, conversion scales, and pawn-valued move features."""

    color = board.turn
    enemy = not color
    before = before_context or board_context(board, scales)
    after = after_board
    if after is None:
        after = board.copy(stack=False)
        after.push(move)
    after_values = after_context or board_context(after, scales)

    castling_event, rights_event = castling_transition_raw(board, move, before, after_values)
    raw = {
        "material": after_values.material_for_side(color) - before.material_for_side(color),
        "center": after_values.center_raw_by_side[color] - before.center_raw_by_side[color],
        "development": float(
            after_values.development_slots_by_side[color]
            - before.development_slots_by_side[color]
        ),
        "castling": castling_event + rights_event,
        "king_safety": (
            after_values.king_safety_raw_by_side[color]
            - before.king_safety_raw_by_side[color]
        ),
        "king_pressure": float(
            king_safe_squares(board, enemy, color)
            - king_safe_squares(after, enemy, color)
        ),
    }
    scale = {
        "material": 1.0,
        "center": scales.center_control,
        "development": scales.development_per_minor,
        "castling": 1.0,
        "king_safety": scales.king_safety,
        "king_pressure": scales.king_freedom,
    }
    pawn = {
        "material": raw["material"],
        "center": scale["center"] * raw["center"],
        "development": scale["development"] * raw["development"],
        "castling": scales.castling_move * castling_event + scales.castling_right * rights_event,
        "king_safety": scale["king_safety"] * raw["king_safety"],
        "king_pressure": scale["king_pressure"] * raw["king_pressure"],
    }
    return {
        "raw": raw,
        "scales": scale,
        "pawn": pawn,
        "castling_raw_events": {
            "castle": castling_event,
            "rights": rights_event,
        },
        "castling_scales": {
            "castle": scales.castling_move,
            "rights": scales.castling_right,
        },
        "game_phase": before.game_phase,
    }


def move_features(
    board: chess.Board,
    move: chess.Move,
    before_context: BoardFeatureContext | None = None,
    after_context: BoardFeatureContext | None = None,
    after_board: chess.Board | None = None,
    scales: FeatureScales = DEFAULT_SCALES,
) -> Dict[str, float]:
    """Six pawn-valued features from the mover's perspective."""

    breakdown = move_feature_breakdown(
        board,
        move,
        before_context=before_context,
        after_context=after_context,
        after_board=after_board,
        scales=scales,
    )
    pawn = breakdown["pawn"]
    return {
        "material": pawn["material"],
        "center": pawn["center"],
        "development": pawn["development"],
        "castling": pawn["castling"],
        "king_safety": pawn["king_safety"],
        "king_pressure": pawn["king_pressure"],
        "game_phase": float(breakdown["game_phase"]),
    }


def white_minus_black_features(
    board: chess.Board, context: BoardFeatureContext | None = None
) -> Dict[str, float]:
    values = context or board_context(board)
    return {
        "material": values.material_balance,
        "pawn_structure": (
            values.pawn_structure_by_side[chess.WHITE]
            - values.pawn_structure_by_side[chess.BLACK]
        ),
        "mobility": (
            values.mobility_score_by_side[chess.WHITE]
            - values.mobility_score_by_side[chess.BLACK]
        ),
        "center": values.center_by_side[chess.WHITE] - values.center_by_side[chess.BLACK],
        "king_safety": (
            values.king_safety_by_side[chess.WHITE]
            - values.king_safety_by_side[chess.BLACK]
        ),
    }
