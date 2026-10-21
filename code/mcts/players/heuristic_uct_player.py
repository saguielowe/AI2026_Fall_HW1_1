"""Ready-to-use player for heuristic-assisted UCT MCTS."""

from collections.abc import Callable

from env.base_env import BaseGame
from mcts.heuristic_mcts import HeuristicUCTMCTS, HeuristicUCTMCTSConfig
from other_algo.heuristic import go_heuristic_evaluation
from .uct_player import UCTPlayer


class HeuristicUCTPlayer(UCTPlayer):
    """UCT player that evaluates non-terminal leaves with a heuristic."""

    def __init__(
        self,
        config: HeuristicUCTMCTSConfig = None,
        evaluator: Callable[[BaseGame], float] = go_heuristic_evaluation,
        deterministic: bool = False,
    ) -> None:
        if config is None:
            config = HeuristicUCTMCTSConfig()
        super().__init__(config=config, deterministic=deterministic)
        self.evaluator = evaluator

    def __str__(self):
        return f"Heuristic UCT Player (weight={self.config.heuristic_weight:g})"

    def init(self, init_game: BaseGame):
        self.mcts = HeuristicUCTMCTS(
            init_game,
            self.config,
            self.evaluator,
        )
