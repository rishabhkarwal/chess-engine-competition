import sys
import os
import ctypes
import numpy as np
from numba import cfunc, types, carray
from evaluation import evaluation_function
from rich.console import Console

_console = Console()

def log(text : str):
    _console.log(text)

BRAIN_OFFSET = 0x4EE0 # magic offset for search function (windows)
COMPETITION_DEPTH = 5

# the c structures in python
class SearchContext(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('depth', ctypes.c_int32),
        ('padding', ctypes.c_int32),
        ('callback_ptr', ctypes.c_void_p) # pointer to the python function
    ]


@cfunc(types.int32(types.CPointer(types.int64), types.CPointer(types.int64), types.uint32))
def evaluation_wrapper(board_pieces_ptr, board_occupancy_ptr, side_to_move):

    board_pieces = carray(board_pieces_ptr, (12,), np.int64)
    board_occupancy = carray(board_occupancy_ptr, (3,), np.int64)
    
    score = evaluation_function(board_pieces, board_occupancy, side_to_move)
    
    return np.int32(score)

def move_to_str(move : int) -> str:
    # parse output (has been bit-packed)
    from_sq = move & 0x3F
    to_sq = (move >> 6) & 0x3F
    if from_sq == to_sq: return '0000'
    files = 'abcdefgh'
    return f'{files[from_sq % 8]}{from_sq // 8 + 1}{files[to_sq % 8]}{to_sq // 8 + 1}'

def setup_board():
    # setup the board (start position)
    BoardArrayType = ctypes.c_uint64 * 32 # works at 17 but not any lower ? 
    board = BoardArrayType()

    # initialise white pieces
    board[0] = 0xFF00; board[1] = 0x42; board[2] = 0x24; 
    board[3] = 0x81;   board[4] = 0x08; board[5] = 0x10;
    # initialise black pieces
    board[6] = 0xFF000000000000; board[7] = 0x4200000000000000; board[8] = 0x2400000000000000;
    board[9] = 0x8100000000000000; board[10]= 0x0800000000000000; board[11]= 0x1000000000000000;
    
    # calculate occupancy
    white = 0
    black = 0
    for i in range(6): white |= board[i]
    for i in range(6, 12): black |= board[i]
    
    board[12] = white
    board[13] = black
    board[14] = white | black

    return board

def get_best_move(depth=5):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(curr_dir, '..', 'bindings', 'ChessLib.dll')
    
    try:
        dll = ctypes.CDLL(lib_path)
    except OSError:
        log('[bold red]Error:[/bold red] could not load ChessLib.dll')
        return '0000'

    # calculate the hidden function address
    base_address = dll._handle # base address of the loaded module in memory
    brain_address = base_address + BRAIN_OFFSET

    # define the function signature
    # 'short* (__fastcall *Brain)(short* out, uint64* board, Context* ctx, int* stats)'
    BrainFuncType = ctypes.CFUNCTYPE(
        ctypes.POINTER(ctypes.c_int16), # return type (short*)
        ctypes.POINTER(ctypes.c_int16), # arg 1: output move
        ctypes.POINTER(ctypes.c_uint64), # arg 2: board array
        ctypes.POINTER(SearchContext), # arg 3: context struct
        ctypes.POINTER(ctypes.c_int32) # arg 4: stats array
    )
    
    # create the callable object from the address
    run_search = BrainFuncType(brain_address)

    board = setup_board()

    # setup call arguments
    best_move_out = ctypes.c_int16(0)
    stats_out = (ctypes.c_int32 * 4)()
    
    ctx = SearchContext()
    ctx.depth = depth
    ctx.padding = 0
    ctx.callback_ptr = ctypes.cast(evaluation_wrapper.address, ctypes.c_void_p)

    log(f'go startpos depth {depth}')
    run_search(ctypes.byref(best_move_out), board, ctypes.byref(ctx), stats_out)
    
    return move_to_str(best_move_out.value)
    
if __name__ == '__main__':
    move = get_best_move(depth=5)
    print(f'best move {move}')