import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualizer import ExperimentManager, GameSession


def test_go_human_move_and_pass():
    session = GameSession()
    session.new_game({"size": 5, "black": "human", "white": "human"})
    session.human_move(0); session.human_move(25)
    state = session.state()
    assert state["game"] == "go"
    assert state["board"][0][0] == 1
    assert state["history"][1]["label"] == "停一手"


def test_real_mcts_search_produces_replay_and_applies_move():
    session = GameSession()
    session.new_game({"size": 5, "black": "mcts", "white": "human", "n_search": 3, "n_rollout": 1})
    payload = session.prepare_search()
    phases = [event["phase"] for event in session.trace]
    assert payload["trace_count"] >= 2 and phases[0] == "start"
    result = session.skip_search()
    assert result["state"]["error"] is None
    assert len(result["state"]["history"]) == 1


def test_alpha_beta_is_restricted_to_5_by_5():
    session = GameSession()
    with pytest.raises(ValueError, match="5×5"):
        session.new_game({"size": 7, "black": "alphabeta", "white": "human"})
    state = session.new_game({"size": 5, "black": "alphabeta", "white": "human", "ab_depth": 2})
    assert state["config"]["ab_depth"] == 2
    with pytest.raises(ValueError, match="1 到 3"):
        session.new_game({"size": 5, "black": "alphabeta", "white": "human", "ab_depth": 4})


def test_heuristic_mcts_is_available_in_single_game_ui_backend():
    session = GameSession()
    state = session.new_game({
        "size": 5, "black": "heuristic_mcts", "white": "human",
        "n_search": 2, "n_rollout": 1, "heuristic_weight": 0.7,
        "heuristic_scale": 2.0,
    })
    assert state["players"]["black"] == "heuristic_mcts"
    assert session.players[1].config.heuristic_weight == 0.7


def test_batch_experiment_runs_in_background_and_swaps_colors():
    manager = ExperimentManager()
    status = manager.start({
        "size": 5, "games": 1, "swap": True, "seed": 2026,
        "player_a": {"algorithm": "random"},
        "player_b": {"algorithm": "random"},
    })
    assert status["total"] == 2
    manager.worker.join(timeout=5)
    result = manager.status()
    assert not result["running"]
    assert result["error"] is None
    assert result["completed"] == 2
    assert len(result["rows"]) == 2
    assert result["last_game"]["size"] == 5


def test_batch_experiment_rejects_alpha_beta_on_larger_board():
    manager = ExperimentManager()
    with pytest.raises(ValueError, match="5×5"):
        manager.start({
            "size": 7, "games": 1,
            "player_a": {"algorithm": "alphabeta"},
            "player_b": {"algorithm": "random"},
        })


def test_backup_trace_marks_every_updated_node():
    session = GameSession()
    session.new_game({"size": 5, "black": "mcts", "white": "human", "n_search": 4, "n_rollout": 1})
    session.prepare_search()
    backups = [event for event in session.trace if event["phase"] == "backup"]
    assert backups
    for event in backups:
        assert {change["id"] for change in event["backup_changes"]} == set(event["path"])
        assert all(change["delta_n"] == 1 for change in event["backup_changes"])


def test_new_game_cancels_running_mcts():
    session = GameSession()
    session.new_game({"size": 5, "black": "mcts", "white": "human", "n_search": 30, "n_rollout": 1})
    player = session.players[1]; player.init(session.game)
    original_rollout = player.mcts.rollout
    def slow_rollout(_mcts, node):
        time.sleep(0.02)
        return original_rollout(node)
    player.mcts.rollout = types.MethodType(slow_rollout, player.mcts)
    worker = threading.Thread(target=session.prepare_search); worker.start()
    while not session.search_progress()["running"]:
        time.sleep(0.001)
    state = session.new_game({"size": 5, "black": "human", "white": "human"})
    worker.join(timeout=0.6)
    assert not worker.is_alive()
    assert state["history"] == [] and np.count_nonzero(state["board"]) == 0
    assert not session._active_searches
