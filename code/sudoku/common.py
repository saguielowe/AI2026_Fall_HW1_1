"""4x4 数独的公共定义和工具函数。"""

SIZE = 4
BOX_SIZE = 2
DIGITS = tuple(range(1, SIZE + 1))


def create_empty_board():
    """创建全为 0 的 4x4 棋盘。"""
    return [[0] * SIZE for _ in range(SIZE)]


def copy_board(board):
    """返回棋盘的独立副本。"""
    return [row[:] for row in board]


def print_board(board):
    """在终端中打印棋盘。"""
    for row in board:
        print(" ".join(str(value) for value in row))


def is_valid_solution(board):
    """检查棋盘是否为完整合法的 4x4 数独解。"""
    expected = set(DIGITS)
    if len(board) != SIZE or any(len(row) != SIZE for row in board):
        return False
    if any(set(row) != expected for row in board):
        return False
    if any({board[row][col] for row in range(SIZE)} != expected for col in range(SIZE)):
        return False

    for box_row in range(0, SIZE, BOX_SIZE):
        for box_col in range(0, SIZE, BOX_SIZE):
            values = {
                board[row][col]
                for row in range(box_row, box_row + BOX_SIZE)
                for col in range(box_col, box_col + BOX_SIZE)
            }
            if values != expected:
                return False
    return True
