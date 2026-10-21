"""完成 TODO 后可用于自测的基础测试。"""

try:
    from .backtracking import backtrack, is_valid
    from .common import DIGITS, create_empty_board, is_valid_solution
    from .local_search import get_conflicts, initialize_board, min_conflicts
except ImportError:
    from backtracking import backtrack, is_valid
    from common import DIGITS, create_empty_board, is_valid_solution
    from local_search import get_conflicts, initialize_board, min_conflicts


EXPECTED_SOLUTION = [
    [1, 2, 3, 4],
    [3, 4, 1, 2],
    [2, 1, 4, 3],
    [4, 3, 2, 1],
]


def assert_valid_solution(board):
    assert is_valid_solution(board)


def test_is_valid():
    board = create_empty_board()
    assert is_valid(board, 0, 0, 1)

    row_conflict = [[1, 0, 0, 0] for _ in range(4)]
    assert not is_valid(row_conflict, 0, 1, 1)

    column_conflict = [[0, 0, 0, 0] for _ in range(4)]
    column_conflict[2][0] = 1
    assert not is_valid(column_conflict, 0, 0, 1)

    box_conflict = [[0, 0, 0, 0] for _ in range(4)]
    box_conflict[1][1] = 1
    assert not is_valid(box_conflict, 0, 0, 1)


def test_backtracking():
    board = create_empty_board()
    solutions = []
    backtrack(board, solutions)
    assert len(solutions) == 288
    assert EXPECTED_SOLUTION in solutions
    assert all(assert_valid_solution(solution) is None for solution in solutions)
    assert board == create_empty_board()  # 检查回溯后状态是否完整恢复


def test_initialize_board():
    board = initialize_board()
    assert all(set(row) == set(DIGITS) for row in board)


def test_get_conflicts():
    board = [row[:] for row in EXPECTED_SOLUTION]
    assert get_conflicts(board, 0, 0) == 0

    board[1] = [1, 2, 3, 4]
    assert get_conflicts(board, 0, 0) == 1  # 同一冲突不能因“同列+同宫”被计两次


def test_min_conflicts():
    solution = min_conflicts(max_steps=1000, max_restarts=20)
    assert solution is not None
    assert_valid_solution(solution)
