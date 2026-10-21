"""不启动浏览器，测试分步搜索代码与可视化会话的连接。"""

import random

try:
    from .local_search import DIGITS, min_conflicts_steps
    from .visualizer import SearchSession
except ImportError:
    from local_search import DIGITS, min_conflicts_steps
    from visualizer import SearchSession


def test_generator_emits_three_micro_steps():
    events = min_conflicts_steps(rng=random.Random(0))
    initialize = next(events)
    assert initialize["phase"] == "initialize"

    selected = next(events)
    assert selected["phase"] == "select"
    selected_coord = selected["selected"]

    evaluated = next(events)
    assert evaluated["phase"] == "evaluate"
    assert len(evaluated["candidates"]) == 3

    executed = next(events)
    assert executed["phase"] == "execute"
    before = evaluated["board"]
    after = executed["board"]
    changed_rows = [row for row in range(4) if before[row] != after[row]]
    assert changed_rows == [selected_coord[0]]
    assert all(set(row) == set(DIGITS) for row in after)


def test_visualizer_displays_generator_output_without_replacing_it():
    custom_board = [
        [4, 3, 2, 1],
        [1, 2, 3, 4],
        [2, 1, 4, 3],
        [3, 4, 1, 2],
    ]

    def fake_generator(**_kwargs):
        yield {
            "phase": "initialize",
            "board": custom_board,
            "total_conflicts": 123,
            "message": "custom-output",
        }
        yield {
            "phase": "failed",
            "board": custom_board,
            "total_conflicts": 456,
            "message": "custom-second-output",
        }

    session = SearchSession(generator_factory=fake_generator)
    assert session.state["board"] == custom_board
    assert session.state["total_conflicts"] == 123
    assert session.state["message"] == "custom-output"

    state = session.advance()
    assert state["board"] == custom_board
    assert state["total_conflicts"] == 456
    assert state["message"] == "custom-second-output"


def test_generator_reaches_a_solution():
    events = min_conflicts_steps(rng=random.Random(7))
    last = None
    for _ in range(5000):
        last = next(events)
        if last.get("solved"):
            break
    assert last is not None and last.get("solved")
    assert last["total_conflicts"] == 0
    assert all(set(row) == set(DIGITS) for row in last["board"])
