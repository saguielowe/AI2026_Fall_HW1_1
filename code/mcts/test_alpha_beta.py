import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.go.go_env import GoGame
from other_algo.alpha_beta_search import AlphaBetaSearch
from other_algo.heuristic import go_heuristic_evaluation


def test_depth_two_alpha_beta_returns_a_legal_action_and_emits_updates():
    game = GoGame(5); game.reset()
    search = AlphaBetaSearch(depth=2)
    events = []
    search.trace_callback = events.append
    action = search.get_best_move(game)
    assert game.action_mask[action] == 1
    phases = {event["phase"] for event in events}
    assert {"node", "leaf", "update", "finish"} <= phases


def test_go_heuristic_uses_the_current_player_perspective():
    game = GoGame(5)
    game.reset()
    game.step(12)
    white_view = go_heuristic_evaluation(game)

    black_to_move = game.fork()
    black_to_move._current_player = 1
    black_view = go_heuristic_evaluation(black_to_move)

    assert black_view > 0
    assert np.isclose(white_view, -black_view)
