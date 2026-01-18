from numba import njit, int64, int32
import numpy as np

"""Piece IDs"""
WHITE_PAWN   = 0
WHITE_KNIGHT = 1
WHITE_BISHOP = 2
WHITE_ROOK   = 3
WHITE_QUEEN  = 4
WHITE_KING   = 5

BLACK_PAWN   = 6
BLACK_KNIGHT = 7
BLACK_BISHOP = 8
BLACK_ROOK   = 9
BLACK_QUEEN  = 10
BLACK_KING   = 11

EMPTY = 12

@njit(int32(int64[:], int32))
def get_piece(board_pieces, sq):
    """
    Returns the Piece ID at a specific square (0-63).
    """
    mask = 1 << sq

    # loop through the 12 bitboards (0-5 White, 6-11 Black)
    for piece_idx in range(12):
        if board_pieces[piece_idx] & mask:
            return piece_idx

    return EMPTY

@njit(int32(int64[:], int32, int32))
def check_square(board_occupancy, sq, color_idx):
    """
    Checks if a specific color occupies a square
    color_idx: 0 for White, 1 for Black, 2 for B&W
    """
    mask = 1 << sq
    if board_occupancy[color_idx] & mask:
        return 1
    return 0

@njit(int32(int64))
def count_bits(bitboard):
    """
    Counts the number of pieces in a bitboard
    """
    count = 0
    while bitboard:
        bitboard &= bitboard - 1
        count += 1
    return count