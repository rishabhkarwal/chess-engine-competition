import board_tools as bt
from numba import njit, int64, int32, uint32

# piece values
PAWN   = 100
KNIGHT = 320
BISHOP = 330
ROOK   = 500
QUEEN  = 900
KING   = 20000

# piece-square tables
# NOTE: defined from white's perspective

# pawn: encourage pushing to centre and promotion
pawn_table = (
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
)

# knight: strong in centre, weak in corners
knight_table = (
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
)

# bishop: avoid corners, control diagonals
bishop_table = (
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
)

# rook: loves 7th rank and centre files
rook_table = (
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0
)

# queen: slight preference for centre, keep mobile
queen_table = (
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,   0,  5,  5,  5,  5,  0, -5,
     0,   0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
)

# king: hide in castle during middlegame (safety)
king_table = (
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
)

@njit(int32(int64[:], int64[:], uint32))
def evaluation_function(board_pieces, board_occupancy, side_to_move):
    score = 0

    for sq in range(64):
        piece = bt.get_piece(board_pieces, sq)
        
        if piece == 0: continue

        # white
        if piece == bt.WHITE_PAWN: score += PAWN + pawn_table[sq]
        elif piece == bt.WHITE_KNIGHT: score += KNIGHT + knight_table[sq]
        elif piece == bt.WHITE_BISHOP: score += BISHOP + bishop_table[sq]
        elif piece == bt.WHITE_ROOK: score += ROOK + rook_table[sq]
        elif piece == bt.WHITE_QUEEN: score += QUEEN + queen_table[sq]
        elif piece == bt.WHITE_KING: score += KING + king_table[sq]

        # black
        # mirror the square index to read the table from black's perspective
        else:
            sq = sq ^ 56
        
            if piece == bt.BLACK_PAWN: score -= PAWN + pawn_table[sq]
            elif piece == bt.BLACK_KNIGHT: score -= KNIGHT + knight_table[sq]
            elif piece == bt.BLACK_BISHOP: score -= BISHOP + bishop_table[sq]
            elif piece == bt.BLACK_ROOK: score -= ROOK + rook_table[sq]
            elif piece == bt.BLACK_QUEEN: score -= QUEEN + queen_table[sq]
            elif piece == bt.BLACK_KING: score -= KING + king_table[sq]

    if side_to_move == 1: return -score # black
    return score