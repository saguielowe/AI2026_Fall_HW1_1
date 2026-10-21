"""围棋局面评估函数（已提供，不需要填写）。"""

from env.base_env import BLACK, BaseGame


def go_heuristic_evaluation(game_state: BaseGame) -> float:
    """返回当前行动方视角的局面分数；越大表示当前方越有利。

    底层棋盘的 ``get_score()`` 使用与终局一致的简化数子和领地
    规则，返回 ``[黑方分数, 白方分数]``。这里先求黑白分差，
    再根据 ``current_player`` 转成当前行动方视角，最后除以棋盘
    格点数，使不同棋盘大小下的分数尺度基本一致。
    """
    black_score, white_score = game_state.board.get_score()
    black_advantage = float(black_score - white_score)
    current_advantage = (
        black_advantage
        if game_state.current_player == BLACK
        else -black_advantage
    )
    return current_advantage / float(game_state.n * game_state.n)
