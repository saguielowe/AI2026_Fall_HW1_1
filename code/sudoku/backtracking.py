"""4x4 数独：回溯搜索实现。

规则：每行、每列、每个 2x2 小宫格都必须填入 1-4，且不能重复。
0 表示待填写的空格。本题不提供任何初始数字，搜索从全空棋盘开始。
"""

try:
    from .common import BOX_SIZE, DIGITS, SIZE, copy_board, create_empty_board, print_board
except ImportError:
    from common import BOX_SIZE, DIGITS, SIZE, copy_board, create_empty_board, print_board

def is_valid(board, row, col, value):
    """检查在 (row, col) 填入 value 是否合法。

    需要同时检查：
    1. value 是否已经出现在第 row 行；
    2. value 是否已经出现在第 col 列；
    3. value 是否已经出现在对应的 2x2 小宫格。

    参数:
        board: 4x4 二维列表，0 表示空格
        row: 待填写格子的行号
        col: 待填写格子的列号
        value: 候选数字，取值为 1-4

    返回:
        bool: 合法返回 True，否则返回 False
    """
    if value not in DIGITS:
        return False

    # 检查同一行。
    if value in board[row]:
        return False

    # 检查同一列。
    if any(board[other_row][col] == value for other_row in range(SIZE)):
        return False

    # 根据当前坐标找到所在 2x2 小宫格的左上角。
    box_row = (row // BOX_SIZE) * BOX_SIZE
    box_col = (col // BOX_SIZE) * BOX_SIZE
    for current_row in range(box_row, box_row + BOX_SIZE):
        for current_col in range(box_col, box_col + BOX_SIZE):
            if board[current_row][current_col] == value:
                return False

    return True


def find_empty(board):
    """按行优先顺序返回第一个空格坐标，没有空格时返回 None。"""
    for row in range(SIZE):
        for col in range(SIZE):
            if board[row][col] == 0:
                return row, col
    return None


def backtrack(board, solutions):
    """使用回溯法枚举当前 4x4 数独的所有解。

    算法应当：
    1. 找到一个空格；
    2. 依次尝试 1-4；
    3. 合法时写入候选数字并递归；
    4. 递归返回后恢复为 0；
    5. 没有空格时，将当前棋盘的深拷贝加入 solutions。

    参数:
        board: 当前数独状态
        solutions: 用于保存所有合法解的列表
    """
    empty = find_empty(board)

    # 没有空格时，当前棋盘就是一个完整解。
    # 必须保存深拷贝，避免后续回溯修改已保存的解。
    if empty is None:
        solutions.append(copy_board(board))
        return

    row, col = empty
    for value in DIGITS:

        if is_valid(board, row, col, value):
            board[row][col] = value  # 填入候选数字
            backtrack(copy_board(board), solutions)  # 递归搜索
            board[row][col] = 0  # 恢复为空格，尝试其他候选数字
        
        # TODO：完成回溯搜索的关键步骤：
        # 1. 判断 value 是否可以填入 (row, col)；
        # 2. 合法时填入 value，并递归搜索；
        # 3. 递归返回后将该格恢复为 0。
        # 提示：恢复为 0 必须在递归返回后执行，否则无法继续尝试其他分支。


if __name__ == "__main__":
    board = create_empty_board()
    solutions = []
    backtrack(board, solutions)
    print(f"4x4 数独共找到 {len(solutions)} 个解。")
    if solutions:
        print("其中一个解为：")
        print_board(solutions[0])
