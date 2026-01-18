import sys
import os
import chess
import importlib
from tqdm import tqdm
from contextlib import redirect_stdout
from driver import Engine
import math
import random
import time
os.system('') # enable ANSI escape codes on windows

COMPETITION_DEPTH = 5
LOG_FILE = 'history.csv'
START_POSITIONS = 'openings.txt'

class Player:
    def __init__(self, strategy):
        self.strategy = strategy
        self.engine = self.load_engine()
        self.stats = {'W' : 0, 'D' : 0, 'L' : 0} # wins, draws, losses

    def load_engine(self):
        # suppress initialisation prints
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            module = importlib.import_module(f'functions.{self.strategy}')
            return Engine(module.evaluation_function, COMPETITION_DEPTH)

    def get_stats(self):
        return '/'.join(str(result) for result in self.stats.values())

def play_game(white, black, start_fen = None):
    board = chess.Board(start_fen) if start_fen else chess.Board()
    # suppress info printing during game
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        while not board.is_game_over():
            chessboard = str(board)
            sys.stderr.write(f"\033[{chessboard.count('\n') + 2}F")
            sys.stderr.write(str(board) + '\n\n')
            engine = white.engine if board.turn == chess.WHITE else black.engine
            move_str = engine.get_best_move(board)
            board.push(chess.Move.from_uci(move_str))
            
    return board.result()

def save_history(A, B, games):
    if games == 0: return

    # calculate score
    score = A.stats['W'] + (A.stats['D'] * 0.5)
    total = games
    
    # calculate win rate %
    win_rate = score / total
    
    # calculate Elo difference
    clamped_score = max(0.01, min(0.99, win_rate))
    elo_diff = -400 * math.log10(1 / clamped_score - 1)
    
    csv_line = (f'{A.strategy},'
                f'{B.strategy},'
                f'{games},'
                f'{A.stats["W"]},'
                f'{A.stats["D"]},'
                f'{A.stats["L"]},'
                f'{win_rate * 100:.1f}%,'
                f'{elo_diff :+.0f}\n')
    
    # write to file
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a') as f:
        if not file_exists: f.write("New,Old,Games,W,D,L,Win Rate,Elo Difference\n") # create header if missing
        f.write(csv_line)
        
    print(f'\nsaved result to \'{LOG_FILE}\'')

def match(A, B, games=10):
    try:
        played = 0

        openings = []
        if os.path.exists(START_POSITIONS):
            with open(START_POSITIONS, 'r') as f:
                openings = [line.strip() for line in f if line.strip()]

        os.system('cls' if os.name == 'nt' else 'clear')
        sys.stderr.write('\n' * 8 + '\n\n')
        
        pbar = tqdm(range(games), unit="game", dynamic_ncols=True)
        
        for i in pbar:
            # swap sides
            if i & 1:
                white, black = B, A
            else:
                white, black = A, B
            
            fen = random.choice(openings) if openings else None

            pbar.set_description(f"{fen if fen else 'startpos'}")

            # update bar text
            score_str = (f'{white.strategy}: {white.get_stats()} | '
                        f'{black.strategy}: {black.get_stats()}')
            pbar.set_postfix_str(score_str)

            result = play_game(white, black, fen)

            # update stats
            if result == '1/2-1/2':
                white.stats['D'] += 1
                black.stats['D'] += 1
            elif result == '1-0':
                white.stats['W'] += 1; black.stats['L'] += 1
            elif result == '0-1':
                black.stats['W'] += 1; white.stats['L'] += 1
            
            played += 1
            
            # update bar text
            score_str = (f'{white.strategy}: {white.get_stats()} | '
                        f'{black.strategy}: {black.get_stats()}')
            pbar.set_postfix_str(score_str)

    except KeyboardInterrupt:
        pass

    print('\n\n')
    [print(f'{player.strategy} : {player.stats["W"] + player.stats["D"] / 2 :.1f}') for player in [A, B]]

    return played

if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else 'psqt'
    b = sys.argv[2] if len(sys.argv) > 2 else 'material'
    n_games = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    engine_a, engine_b = Player(a), Player(b)
    played = match(engine_a, engine_b, n_games)
    save_history(engine_a, engine_b, played)