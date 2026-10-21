"""UCT MCTS whose leaf value can use a heuristic evaluation function."""

from collections.abc import Callable

import numpy as np

from env.base_env import BaseGame
from .node import MCTSNode
from .uct_mcts import UCTMCTS, UCTMCTSConfig


class HeuristicUCTMCTSConfig(UCTMCTSConfig):
    """Configuration for heuristic-assisted UCT.

    ``heuristic_weight`` controls how a non-terminal leaf is evaluated:

        value = w * heuristic_value + (1 - w) * rollout_value

    Therefore, ``w=1`` replaces rollout with the heuristic and ``0<w<1``
    blends the two estimates. The raw heuristic is mapped to ``[-1, 1]`` by
    ``tanh(raw_value / heuristic_scale)`` so that it is comparable with a
    win/draw/loss rollout value.
    """

    def __init__(
        self,
        heuristic_weight: float = 1.0,
        heuristic_scale: float = 1.0,
        n_rollout: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(n_rollout=n_rollout, *args, **kwargs)
        if not 0.0 <= heuristic_weight <= 1.0:
            raise ValueError("heuristic_weight must be in [0, 1]")
        if heuristic_scale <= 0:
            raise ValueError("heuristic_scale must be positive")
        self.heuristic_weight = float(heuristic_weight)
        self.heuristic_scale = float(heuristic_scale)


class HeuristicUCTMCTS(UCTMCTS):
    """UCT MCTS using an evaluation function at non-terminal leaf nodes.

    The evaluator must return a scalar from the perspective of
    ``game_state.current_player``: a larger value means that the position is
    better for the player who is about to move.
    """

    def __init__(
        self,
        init_env: BaseGame,
        config: HeuristicUCTMCTSConfig,
        evaluator: Callable[[BaseGame], float],
        root: MCTSNode = None,
    ):
        self.evaluator = evaluator
        super().__init__(init_env, config, root)

    def evaluate(self, node: MCTSNode) -> float:
        """Return a bounded heuristic value for ``node``."""
        raw_value = float(self.evaluator(node.env))
        if not np.isfinite(raw_value):
            raise ValueError("the heuristic evaluator must return a finite number")
        return float(np.tanh(raw_value / self.config.heuristic_scale))

    def rollout(self, node: MCTSNode) -> float:
        """Evaluate a leaf by replacement or weighted correction of rollout."""
        heuristic_value = self.evaluate(node)
        weight = self.config.heuristic_weight
        if weight == 1.0:
            return heuristic_value

        rollout_value = super().rollout(node)
        return weight * heuristic_value + (1.0 - weight) * rollout_value

    def search(self):
        """Run UCT without repeating a deterministic replacement evaluation.

        With ``heuristic_weight == 1``, rollout is completely replaced by one
        deterministic heuristic value.  Repeating that same evaluation
        ``n_rollout`` times cannot change the average, so this method temporarily
        uses one evaluation per selected leaf.  The configured value is restored
        afterwards and remains effective when rollout is actually blended in.
        """
        if self.config.heuristic_weight != 1.0 or self.config.n_rollout == 1:
            return super().search()

        configured_rollouts = self.config.n_rollout
        self.config.n_rollout = 1
        try:
            return super().search()
        finally:
            self.config.n_rollout = configured_rollouts

    def get_subtree(self, action: int):
        """Reuse a searched child while preserving evaluator and configuration."""
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return HeuristicUCTMCTS(
                new_root.env,
                self.config,
                self.evaluator,
                root=new_root,
            )
        return None
