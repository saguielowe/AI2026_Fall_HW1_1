from other_algo.alpha_beta_search import AlphaBetaSearch
from .base_player import BasePlayer

class AlphaBetaPlayer(BasePlayer):
    def __init__(self, depth=2):
        self.policy = AlphaBetaSearch(depth=depth)
        self.As = {}

    def __str__(self):
        return "AlphaBeta Player"

    def play(self, state):
        s = state.to_string()
        if s not in self.As:
            self.As[s] = self.policy.get_best_move(state)
        return self.As[s]
