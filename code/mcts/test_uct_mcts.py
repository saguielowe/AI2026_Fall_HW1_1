import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.base_env import EPS
from env.go.go_env import GoGame
from mcts.uct_mcts import UCTMCTS, UCTMCTSConfig


def make_search(n_search=40, n_rollout=2):
    game = GoGame(5)
    game.reset()
    return game, UCTMCTS(game, UCTMCTSConfig(n_search=n_search, n_rollout=n_rollout))


def walk_tree(root):
    stack, seen, max_depth = [(root, 0)], set(), 0
    while stack:
        node, depth = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node)); max_depth = max(max_depth, depth)
        for action, child in node.children.items():
            assert node.action_mask[action]
            assert child.parent is node
            stack.append((child, depth + 1))
    return len(seen), max_depth


def test_search_grows_tree_and_returns_legal_policy():
    np.random.seed(7)
    game, mcts = make_search(n_search=50)
    policy = mcts.search()
    node_count, max_depth = walk_tree(mcts.root)
    assert node_count > game.action_space_size
    assert max_depth >= 2
    assert int(mcts.root.child_N_visit.sum()) == 50
    assert np.isclose(policy.sum(), 1.0)
    assert np.all(policy[game.action_mask == 0] == 0)


def test_search_never_selects_an_occupied_intersection():
    np.random.seed(11)
    game = GoGame(5); game.reset(); game.step(0); game.step(6)
    mcts = UCTMCTS(game, UCTMCTSConfig(n_search=30, n_rollout=2))
    policy = mcts.search()
    assert policy[0] == 0 and policy[6] == 0
    assert np.isclose(policy.sum(), 1.0)


def test_backup_alternates_player_perspective():
    _, mcts = make_search(n_search=1)
    child = mcts.root.add_child(0)
    grandchild = child.add_child(1)
    mcts.backup(grandchild, 0.75)
    assert child.child_N_visit[1] == 1
    assert np.isclose(child.child_V_total[1], -0.75)
    assert mcts.root.child_N_visit[0] == 1
    assert np.isclose(mcts.root.child_V_total[0], 0.75)


def test_draw_reward_is_the_positive_neutral_marker():
    game = GoGame(5)
    game.reset()
    game.step(game.action_space_size - 1)
    _, reward, done = game.step(game.action_space_size - 1)
    assert done
    assert reward == EPS
