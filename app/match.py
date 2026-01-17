import sys
import os
import chess
import importlib
from tqdm import tqdm
from contextlib import redirect_stdout
from driver import Engine

COMPETITION_DEPTH = 5

class Player:
    def __init__(self, strategy):
        self.strategy = strategy
        self.engine = self.load_engine()
        self.stats = {'W' : 0, 'D' : 0, 'L' : 0} # wins, draws, losses

    def load_engine(self):
        # suppress initialisation prints
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            module = importlib.import_module(f'evaluation.{self.strategy}')
            return Engine(module.evaluation_function, COMPETITION_DEPTH)

def play_game(white, black):
    board = chess.Board()
    # suppress move printing during game
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        while not board.is_game_over():
            engine = white.engine if board.turn == chess.WHITE else black.engine
            move_str = engine.get_best_move(board)
            board.push(chess.Move.from_uci(move_str))
            
    return board.result()

def match(A, B, games=10):
    # progress bar
    pbar = tqdm(range(games), unit="game", dynamic_ncols=True)
    
    for i in pbar:
        # swap sides
        if i & 1:
            white, black = B, A
            A_is_white = False
        else:
            white, black = A, B
            A_is_white = True

        result = play_game(white, black)

        # update stats
        if result == '1/2-1/2':
            A.stats['D'] += 1
            B.stats['D'] += 1
        elif result == '1-0':
            if A_is_white:
                A.stats['W'] += 1; B.stats['L'] += 1 # A won
            else:           
                B.stats['W'] += 1; A.stats['L'] += 1 # B won
        elif result == '0-1':
            if A_is_white: 
                B.stats['W'] += 1; A.stats['L'] += 1 # B won
            else:           
                A.stats['W'] += 1; B.stats['L'] += 1 # A won
        
        # update bar text
        score_str = (f'{A.strategy}: {A.stats["W"]}/{A.stats["D"]}/{A.stats["L"]} | '
                     f'{B.strategy}: {B.stats["W"]}/{B.stats["D"]}/{B.stats["L"]}')
        pbar.set_postfix_str(score_str)

    print('\n\n')
    [print(f'{player.strategy} : {player.stats["W"] + player.stats["D"] / 2 :.1f}') for player in [A, B]]

if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else 'material'
    b = sys.argv[2] if len(sys.argv) > 2 else 'constant'
    n_games = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    engine_a, engine_b = Player(a), Player(b)
    match(engine_a, engine_b, n_games)