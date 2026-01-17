import sys
import os
import ctypes
import numpy as np
from numba import cfunc, types, carray
from evaluation import evaluation_function
from rich.console import Console
import chess

_console = Console()

def log(text : str):
    _console.log(text)

BRAIN_OFFSET = 0x4EE0 # magic offset for search function (windows)
COMPETITION_DEPTH = 5

@cfunc(types.int32(types.CPointer(types.int64), types.CPointer(types.int64), types.uint32))
def evaluation_wrapper(board_pieces_ptr, board_occupancy_ptr, side_to_move):

    board_pieces = carray(board_pieces_ptr, (12,), np.int64)
    board_occupancy = carray(board_occupancy_ptr, (3,), np.int64)
    
    score = evaluation_function(board_pieces, board_occupancy, side_to_move)
    
    return np.int32(score)

# the c structures in python
class SearchContext(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('depth', ctypes.c_int32),
        ('padding', ctypes.c_int32),
        ('callback_ptr', ctypes.c_void_p) # pointer to the python function
    ]

class Engine:
    def __init__(self, depth = COMPETITION_DEPTH):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(curr_dir, '..', 'bindings', 'ChessLib.dll')
        
        try:
            dll = ctypes.CDLL(lib_path)
        except OSError:
            log('[bold red]Error:[/bold red] could not load ChessLib.dll')
            return '0000'

        # calculate the hidden function address
        base_address = dll._handle # base address of the loaded module in memory
        self.brain_address = base_address + BRAIN_OFFSET

        # define the function signature
        # 'short* (__fastcall *Brain)(short* out, uint64* board, Context* ctx, int* stats)'
        self.BrainFuncType = ctypes.CFUNCTYPE(
            ctypes.POINTER(ctypes.c_int16), # return type (short*)
            ctypes.POINTER(ctypes.c_int16), # arg 1: output move
            ctypes.POINTER(ctypes.c_uint64), # arg 2: board array
            ctypes.POINTER(SearchContext), # arg 3: context struct
            ctypes.POINTER(ctypes.c_int32) # arg 4: stats array
        )

        self.depth = depth

    @staticmethod
    def move_to_str(move : int) -> str:
        # parse output (has been bit-packed)
        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        if from_sq == to_sq: return '0000'
        files = 'abcdefgh'
        return f'{files[from_sq % 8]}{from_sq // 8 + 1}{files[to_sq % 8]}{to_sq // 8 + 1}'

    @staticmethod
    def get_board(fen : str):
        board = chess.Board(fen)
        arr = (ctypes.c_uint64 * 32)()

        # pieces
        for i, pt in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]):
            arr[i] = board.pieces_mask(pt, chess.WHITE)
            arr[i + 6] = board.pieces_mask(pt, chess.BLACK)

        # occupancy 12-14
        arr[12] = board.occupied_co[chess.WHITE]
        arr[13] = board.occupied_co[chess.BLACK]
        arr[14] = board.occupied

        # state 15-16
        ep = (board.ep_square + 1) if board.ep_square is not None else 0
        arr[15] = (0 if board.turn == chess.WHITE else 1) | (ep << 32)
        
        castling = 0
        if board.has_kingside_castling_rights(chess.WHITE): castling |= 1
        if board.has_queenside_castling_rights(chess.WHITE): castling |= 2
        if board.has_kingside_castling_rights(chess.BLACK): castling |= 4
        if board.has_queenside_castling_rights(chess.BLACK): castling |= 8
        arr[16] = castling

        return arr

    def get_best_move(self, position : str = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'):
        run_search = self.BrainFuncType(self.brain_address)
        
        board = Engine.get_board(position)

        # setup call arguments
        best_move_out = ctypes.c_int16(0)
        stats_out = (ctypes.c_int32 * 4)()
        
        ctx = SearchContext()
        ctx.depth = self.depth
        ctx.padding = 0
        ctx.callback_ptr = ctypes.cast(evaluation_wrapper.address, ctypes.c_void_p)

        log(f'go startpos depth {self.depth}')
        run_search(ctypes.byref(best_move_out), board, ctypes.byref(ctx), stats_out)
        
        return Engine.move_to_str(best_move_out.value)

if __name__ == '__main__':
    engine = Engine()
    move = engine.get_best_move('r1k5/2P2K2/3Rp3/7p/1p4P1/P1P1b3/p1P5/1N2N3 w - - 0 1')
    print(f'best move {move}')