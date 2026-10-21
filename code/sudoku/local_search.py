"""4x4 数独：最小冲突局部搜索实现。"""

import random

try:
    from .common import BOX_SIZE, DIGITS, SIZE, copy_board, print_board
except ImportError:
    from common import BOX_SIZE, DIGITS, SIZE, copy_board, print_board

def get_conflicts(board, row, col):
    """计算 (row, col) 中的数字与其他格子产生多少个冲突。

    由于初始化和后续交换都保证每行恰好包含 1-4，因此只需要
统计同列和同一个 2x2 小宫格中的相同数字。同一个冲突格子
即使同时属于同列和同宫，也只计一次。

    参数:
        board: 当前已填满的 4x4 状态，每行都是 1-4 的一个排列
        row, col: 要评估的格子坐标

    返回:
        int: 冲突格子的数量
    """
    value = board[row][col]
    conflicts = 0

    for other_row in range(SIZE):
        if other_row != row and board[other_row][col] == value:
            conflicts += 1 # 统计同列冲突

    # 根据当前坐标找到所在 2x2 小宫格的左上角。
    box_row = (row // BOX_SIZE) * BOX_SIZE
    box_col = (col // BOX_SIZE) * BOX_SIZE
    for current_row in range(box_row, box_row + BOX_SIZE):
        for current_col in range(box_col, box_col + BOX_SIZE):
            if current_row != row and current_col != col and board[current_row][current_col] == value: # 统计同宫冲突，并排除列冲突已计算的格子
                conflicts += 1

    return conflicts


def initialize_board(rng=None):
    """随机初始化棋盘，保证每行都恰好包含数字 1-4。"""
    rng = rng or random
    return [rng.sample(list(DIGITS), SIZE) for _ in range(SIZE)]


def _total_conflicts(board):
    """统计整个棋盘中不重复的冲突对，供搜索框架复用。"""
    return sum(
        get_conflicts(board, row, col)
        for row in range(SIZE)
        for col in range(SIZE)
    ) // 2


def min_conflicts_steps(max_steps=1000, max_restarts=20, rng=None):
    """逐步生成最小冲突搜索过程，供可视化界面和 min_conflicts 共同使用。

    学生需要使用 yield 依次产生 select、evaluate、execute 三种事件。
    算法应当：
    1. 将每一行初始化为 1-4 的一个随机排列；
    2. 找出所有在列或 2x2 小宫格中产生冲突的格子；
    3. 如果不存在冲突，返回当前棋盘；
    4. 随机选择一个冲突格，尝试将它与同一行的其他三个格子分别交换；
    5. 计算每次交换后整个棋盘的总冲突数，随机执行冲突最少的交换；
    6. 达到 max_steps 仍未解出时重新随机初始化，最多重启 max_restarts 次。

    状态转移限制：不允许直接改写某个格子的数字，唯一允许的操作是
    交换同一行中两个格子的数字。因此搜索过程中每行始终保持 1-4 各一次。

    每个事件是一个字典，至少包含 phase、board、total_conflicts 和 message。
    可视化界面只展示这些事件，不会自行修正学生的计算结果。
    """
    rng = rng or random
    board = initialize_board(rng)
    total_conflicts = _total_conflicts(board)
    yield {
        "phase": "initialize",
        "board": copy_board(board),
        "total_conflicts": total_conflicts,
        "message": "已生成每行均为 1-4 随机排列的初始棋盘。",
        "restart": 0,
        "swaps": 0,
        "solved": total_conflicts == 0,
    }
    if total_conflicts == 0:
        return

    swaps = 0
    for restart in range(max_restarts + 1):
        # restart=0 使用函数开头已生成的棋盘；之后才重新初始化。
        if restart > 0:
            board = initialize_board(rng)
            total_conflicts = _total_conflicts(board)
            yield {
                "phase": "initialize",
                "board": copy_board(board),
                "total_conflicts": total_conflicts,
                "message": f"第 {restart} 次重启：已重新随机初始化棋盘。",
                "restart": restart,
                "swaps": swaps,
                "solved": total_conflicts == 0,
            }
            if total_conflicts == 0:
                return

        for _step in range(max_steps):

            conflict_cells = [
                (row, col)
                for row in range(SIZE)
                for col in range(SIZE)
                if get_conflicts(board, row, col) > 0
            ]
            if not conflict_cells:
                yield {
                    "phase": "solved",
                    "board": copy_board(board),
                    "total_conflicts": total_conflicts,
                    "message": f"已找到一个合法解，使用了 {restart} 次重启和 {swaps} 次交换。",
                    "restart": restart,
                    "swaps": swaps,
                    "solved": True,
                }
                return
            row, col = rng.choice(conflict_cells)
            # TODO 1：收集所有冲突数大于 0 的格子，再用 rng.choice
            # 随机选择一个，并将其行列保存到 row、col。

            yield {
                "phase": "select",
                "board": copy_board(board),
                "total_conflicts": total_conflicts,
                "message": f"选中冲突格：第 {row + 1} 行第 {col + 1} 列。",
                "selected": [row, col],
                "restart": restart,
                "swaps": swaps,
            }

            candidates = []
            for target_col in range(SIZE):
                if target_col == col:
                    continue
                board[row][col], board[row][target_col] = board[row][target_col], board[row][col]
                new_total_conflicts = _total_conflicts(board)
                candidates.append({
                    "target_col": target_col,
                    "total_conflicts": new_total_conflicts,
                })
                board[row][col], board[row][target_col] = board[row][target_col], board[row][col]  # 交换回去
                # 交换后计算总冲突数，并将结果追加到 candidates 列表中。注意交换后要恢复原状

            for candidate in candidates:
                candidate["is_best"] = candidate["total_conflicts"] == min(c["total_conflicts"] for c in candidates)

            chosen_candidates = [c for c in candidates if c["is_best"]]
            chosen_target_col = rng.choice(chosen_candidates)["target_col"]

            yield {
                "phase": "evaluate",
                "board": copy_board(board),
                "total_conflicts": total_conflicts,
                "message": "已评估三个同行交换候选。",
                "selected": [row, col],
                "candidates": candidates,
                "chosen_target_col": chosen_target_col,
                "restart": restart,
                "swaps": swaps,
            }

            board[row][col], board[row][chosen_target_col] = board[row][chosen_target_col], board[row][col]
            swaps += 1
            total_conflicts = _total_conflicts(board)

            solved = total_conflicts == 0
            yield {
                "phase": "execute",
                "board": copy_board(board),
                "total_conflicts": total_conflicts,
                "message": "已执行最优候选交换。",
                "selected": [row, col],
                "chosen_target_col": chosen_target_col,
                "last_swap": [[row, col], [row, chosen_target_col]],
                "restart": restart,
                "swaps": swaps,
                "solved": solved,
            }
            if solved:
                return

    yield {
        "phase": "failed",
        "board": copy_board(board),
        "total_conflicts": total_conflicts,
        "message": "已达到最大重启次数，未找到解。",
        "restart": max_restarts,
        "swaps": swaps,
    }


def min_conflicts(max_steps=1000, max_restarts=20, rng=None):
    """使用与可视化界面完全相同的学生分步实现求解数独。"""
    last_board = None
    for search_event in min_conflicts_steps(max_steps, max_restarts, rng):
        last_board = search_event.get("board")
        if search_event.get("solved"):
            return last_board
    return None


if __name__ == "__main__":
    solution = min_conflicts(max_steps=1000, max_restarts=20)
    if solution:
        print("局部搜索成功找到一个 4x4 数独解：")
        print_board(solution)
    else:
        print("局部搜索未能在限制步数内找到解。")
