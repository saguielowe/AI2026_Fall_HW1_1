import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcts.heuristic_mcts import HeuristicUCTMCTS, HeuristicUCTMCTSConfig
from mcts.uct_mcts import UCTMCTS


class DummyNode:
    def __init__(self, env=None):
        self.env = env if env is not None else object()


def make_search(weight=1.0, scale=1.0, raw_value=2.0):
    search = object.__new__(HeuristicUCTMCTS)
    search.config = HeuristicUCTMCTSConfig(
        heuristic_weight=weight,
        heuristic_scale=scale,
    )
    search.evaluator = lambda env: raw_value
    return search


def test_heuristic_value_is_bounded_and_scaled():
    search = make_search(scale=2.0, raw_value=2.0)
    assert np.isclose(search.evaluate(DummyNode()), np.tanh(1.0))


def test_heuristic_can_replace_rollout(monkeypatch):
    search = make_search(weight=1.0, raw_value=0.5)

    def rollout_must_not_run(self, node):
        raise AssertionError("ordinary rollout should not run in replacement mode")

    monkeypatch.setattr(UCTMCTS, "rollout", rollout_must_not_run)
    assert np.isclose(search.rollout(DummyNode()), np.tanh(0.5))


def test_heuristic_can_correct_rollout(monkeypatch):
    search = make_search(weight=0.25, raw_value=0.0)
    monkeypatch.setattr(UCTMCTS, "rollout", lambda self, node: -0.8)
    assert np.isclose(search.rollout(DummyNode()), -0.6)


def test_replacement_mode_evaluates_each_selected_leaf_only_once(monkeypatch):
    search = make_search(weight=1.0, raw_value=0.5)
    search.config.n_rollout = 7

    observed = []
    monkeypatch.setattr(
        UCTMCTS,
        "search",
        lambda self: observed.append(self.config.n_rollout) or "policy",
    )

    assert search.search() == "policy"
    assert observed == [1]
    assert search.config.n_rollout == 7


def test_invalid_heuristic_config_is_rejected():
    with pytest.raises(ValueError):
        HeuristicUCTMCTSConfig(heuristic_weight=1.1)
    with pytest.raises(ValueError):
        HeuristicUCTMCTSConfig(heuristic_scale=0)
