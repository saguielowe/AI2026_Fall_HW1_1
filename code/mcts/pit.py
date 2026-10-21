from env import *
from players import *
from mcts.uct_mcts import UCTMCTSConfig
from mcts.heuristic_mcts import HeuristicUCTMCTSConfig
from tqdm import trange
from time import perf_counter

import numpy as np

def print_devide_line(n=50):
    print("--"*n)

def pit(game:BaseGame, player1:BasePlayer, player2:BasePlayer, log_output:bool=True):
    game.reset()
    if log_output:
        print(f"start playing {type(game)}")
        print_devide_line()
    reward = 0
    
    for player in [player1, player2]:
        if isinstance(player, UCTPlayer):
            player.clear()
            
    while True:
        a1 = player1.play(game)
        _, reward, done = game.step(a1)
        if isinstance(player2, UCTPlayer):
            player2.opp_play(a1)
        if log_output:
            print(f"Player 1 ({player1}) move: {a1}")
            print(game.to_string())
            print_devide_line()
        if done:
            break
        a2 = player2.play(game)
        _, reward, done = game.step(a2)
        if isinstance(player1, UCTPlayer):
            player1.opp_play(a2)
        if log_output:
            print(f"Player 2 ({player2}) move: {a2}")
            print(game.to_string())
            print_devide_line()
        if done:
            reward *= -1
            break
    if log_output:
        if reward == 1:
            print(f"Player 1 ({player1}) win")
        elif reward == -1:
            print(f"Player 2 ({player2}) win")
        else:
            print("Draw")
    return reward
        
def multi_match(game:BaseGame, player1:BasePlayer, player2:BasePlayer, n_match=100):
    print(f"Player 1:{player1}  Player 2:{player2}")
    n_p1_win, n_p2_win, n_draw = 0, 0, 0
    start_time = perf_counter()
    T = trange(n_match)
    for _ in T:
        reward = pit(game, player1, player2, log_output=False)
        if reward == 1:
            n_p1_win += 1
        elif reward == -1:
            n_p2_win += 1
        else:
            n_draw += 1
        T.set_description_str(f"P1 win: {n_p1_win} ({n_p1_win}) P2 win: {n_p2_win} ({n_p2_win}) Draw: {n_draw} ({n_draw})")        
    elapsed = perf_counter() - start_time
    print(f"Player 1 ({player1}) win: {n_p1_win} ({n_p1_win/n_match*100:.2f}%)")
    print(f"Player 2 ({player2}) win: {n_p2_win} ({n_p2_win/n_match*100:.2f}%)")
    print(f"Draw: {n_draw} ({n_draw/n_match*100:.2f}%)")
    print(f"Player 1 not lose: {n_p1_win+n_draw} ({(n_p1_win+n_draw)/n_match*100:.2f}%)")
    print(f"Player 2 not lose: {n_p2_win+n_draw} ({(n_p2_win+n_draw)/n_match*100:.2f}%)")
    print(f"Elapsed time: {elapsed:.3f}s ({elapsed/n_match:.3f}s per match)")
    return n_p1_win, n_p2_win, n_draw
        
        
def search_best_C():
    from matplotlib import pyplot as plt
    p2nl = []
    cs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0]
    n_match = 100
    for c in cs:
        config = UCTMCTSConfig()
        config.C = c
        config.n_rollout = 7
        config.n_search = 64
        player1 = RandomPlayer()
        player2 = UCTPlayer(config, deterministic=True)
        game = GoGame(5)
        p1w, p2w, drw = multi_match(game, player1, player2, n_match=n_match)
        p2nl.append((p2w+drw)/n_match)
    plt.plot(cs, p2nl)
    plt.savefig('p2nl.png')
        
if __name__ == '__main__':
    #####################
    # Modify code below #
    #####################
    
    # set seed to reproduce the result
    # np.random.seed(0)
        
    game = GoGame(5)
    
    # config for MCTS
    config = UCTMCTSConfig()
    # config.C = 0.5
    # config.C = 0.7
    config.C = 1.0
    # config.C = 2.5
    # config.C = 5.0
    config.n_rollout = 7
    # config.n_rollout = 13
    config.n_search = 64
    # config.n_search = 400

    # 带评估函数的 MCTS（用于围棋）
    # heuristic_weight=1.0：用评估函数完全替代 rollout
    # 0<heuristic_weight<1：融合评估函数与 rollout
    heuristic_config = HeuristicUCTMCTSConfig(
        C=1.0,
        n_search=64,
        n_rollout=1,
        heuristic_weight=1.0,
        heuristic_scale=1.0,
    )
    
    # player initialization    
    # player1 = HumanPlayer()
    player1 = RandomPlayer()
    # Alpha-Beta 只允许用于 GoGame(5)，深度建议从 1 或 2 开始。
    # player1 = AlphaBetaPlayer(depth=2)
    # player1 = UCTPlayer(config, deterministic=True)
    # player1 = HeuristicUCTPlayer(heuristic_config, deterministic=True)
    
    # player2 = HumanPlayer()
    # player2 = RandomPlayer()
    # player2 = AlphaBetaPlayer(depth=2)
    player2 = UCTPlayer(config, deterministic=True)
    # player2 = HeuristicUCTPlayer(heuristic_config, deterministic=True)
    
    # single match
    # pit(game, player1, player2)
    
    multi_match(game, player1, player2, n_match=10)
    multi_match(game, player2, player1, n_match=10)
    # multi_match(game, player1, player2, n_match=100)
    
    #####################
