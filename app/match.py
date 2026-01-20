import sys
import os
import chess
import importlib
import math
import random
import multiprocessing
import numpy as np
import time
from tqdm import tqdm
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed

os.system('') 

COMPETITION_DEPTH = 4
LOG_FILE = 'history.csv'
START_POSITIONS = 'openings.txt'

def play_game_worker(white_strategy, black_strategy, start_fen, depth):
    """Plays a single game in a separate process"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path: sys.path.append(current_dir)
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path: sys.path.append(parent_dir)

    try:
        board = chess.Board(start_fen) if start_fen else chess.Board()
        
        w_module = importlib.import_module(f'functions.{white_strategy}')
        b_module = importlib.import_module(f'functions.{black_strategy}')
        
        try: from driver import Engine
        except ImportError: from app.driver import Engine

        # JIT warmup
        dummy_p = np.zeros(12, dtype=np.int64)
        dummy_o = np.zeros(3, dtype=np.int64)
        w_module.evaluation_function(dummy_p, dummy_o, 0)
        b_module.evaluation_function(dummy_p, dummy_o, 0)

        with open(os.devnull, 'w') as f, redirect_stdout(f):
            white_engine = Engine(w_module.evaluation_function, depth)
            black_engine = Engine(b_module.evaluation_function, depth)

            while not board.is_game_over():
                if board.turn == chess.WHITE:
                    move = white_engine.get_best_move(board)
                else:
                    move = black_engine.get_best_move(board)
                
                board.push(chess.Move.from_uci(move))
            
        return board.result()

    except KeyboardInterrupt:
        os._exit(0)
    except Exception as e:
        return f"[CRASH] {e}"

class PlayerStats:
    def __init__(self, strategy):
        self.strategy = strategy
        self.stats = {'W': 0, 'D': 0, 'L': 0}

    def get_stats(self):
        return '/'.join(str(x) for x in self.stats.values())

def save_history(A, B, games):
    if games == 0: return

    score = A.stats['W'] + (A.stats['D'] * 0.5)
    win_rate = score / games
    clamped_score = max(0.01, min(0.99, win_rate))
    elo_diff = -400 * math.log10(1 / clamped_score - 1)
    
    csv_line = (f'{A.strategy},{B.strategy},{games},'
                f'{A.stats["W"]},{A.stats["D"]},{A.stats["L"]},'
                f'{win_rate * 100:.1f}%,{elo_diff :+.0f}\n')
    
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a') as f:
        if not file_exists: 
            f.write("New,Old,Games,W,D,L,Win Rate,Elo Difference\n")
        f.write(csv_line)
        
    print(f'\nSaved result to \'{LOG_FILE}\'')

def match(strat_a, strat_b, games=10):
    A = PlayerStats(strat_a)
    B = PlayerStats(strat_b)

    openings = []
    if os.path.exists(START_POSITIONS):
        with open(START_POSITIONS, 'r') as f:
            openings = [line.strip() for line in f if line.strip()]

    tasks = []
    for i in range(games):
        fen = random.choice(openings) if openings else None

        if i & 1:
            tasks.append((i, strat_b, strat_a, fen, COMPETITION_DEPTH))
        else:
            tasks.append((i, strat_a, strat_b, fen, COMPETITION_DEPTH))

    played = 0
    errors = 0
    workers = max(1, os.cpu_count() - 1)
    
    print(f"\n{games} games on {workers} cores")
    print(f"{strat_a} vs {strat_b}\n")

    executor = ProcessPoolExecutor(max_workers=workers)

    try:
        futures = {
            executor.submit(play_game_worker, t[1], t[2], t[3], t[4]): t 
            for t in tasks
        }

        pbar = tqdm(as_completed(futures), total=games, unit="game", dynamic_ncols=True)
        
        for future in pbar:
            result = future.result()
            game_info = futures[future]
            game_idx = game_info[0]
            
            if result.startswith("[CRASH]"):
                errors += 1
                pbar.set_description(f"ERROR: {result.split(':')[-1].strip()[:15]}")
                continue

            if game_idx & 1:
                white, black = B, A
            else:
                white, black = A, B
                
            if result == '1/2-1/2': white.stats['D'] += 1; black.stats['D'] += 1
            elif result == '1-0': white.stats['W'] += 1; black.stats['L'] += 1
            elif result == '0-1': black.stats['W'] += 1; white.stats['L'] += 1
            
            played += 1
            pbar.set_postfix_str(f'{A.strategy}: {A.get_stats()} | {B.strategy}: {B.get_stats()}')

    except KeyboardInterrupt:
        print("\n\n[!] interrupted")
        
        executor.shutdown(wait=False, cancel_futures=True)
        
        for p in [A, B]:
            score = p.stats["W"] + p.stats["D"] / 2
            print(f'{p.strategy} : {score:.1f}')
        
        save_history(A, B, played)

        os._exit(0)
    
    executor.shutdown(wait=True)

    print('\n')
    if errors > 0:
        print(f"[WARNING] {errors} games crashed")

    for p in [A, B]:
        score = p.stats["W"] + p.stats["D"] / 2
        print(f'{p.strategy} : {score:.1f}')

    save_history(A, B, played)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    a = sys.argv[1] if len(sys.argv) > 1 else 'mobility'
    b = sys.argv[2] if len(sys.argv) > 2 else 'tapered_eval'
    n_games = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    
    match(a, b, n_games)

   