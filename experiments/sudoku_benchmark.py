"""Independent HW3 comparison. Run: .venv\Scripts\python experiments/sudoku_benchmark.py

Local methods share initial boards, conflict objective, seeds and restart budget.
Time includes baseline's visualization snapshots; objective evaluations are also
reported, and SA proposals must not be equated with accepted swaps.
"""
import csv
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code' / 'sudoku'))
import local_search as baseline
from backtracking import backtrack
from common import is_valid_solution


def anneal(seed, evaluate, budget=1000, restarts=20):
    rng = random.Random(seed)
    proposals = swaps = 0
    first = None
    for restart in range(restarts + 1):
        board = baseline.initialize_board(rng)
        first = first or [row[:] for row in board]
        energy = evaluate(board)
        for step in range(budget):
            if energy == 0:
                return board, proposals, swaps, restart, first
            # Uniformly choose one of the 24 within-row unordered exchanges.
            row = rng.randrange(4)
            c, d = rng.sample(range(4), 2)
            board[row][c], board[row][d] = board[row][d], board[row][c]
            new_energy = evaluate(board)
            temperature = 2.0 * (0.05 / 2.0) ** (step / (budget - 1))
            delta = new_energy - energy
            proposals += 1
            if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                energy = new_energy
                swaps += 1
            else:
                board[row][c], board[row][d] = board[row][d], board[row][c]
        if energy == 0:
            return board, proposals, swaps, restart, first
    return None, proposals, swaps, restart, first


def enumerate_mrv():
    """Recompute candidate domains; propagate singleton domains, otherwise MRV."""
    board = [[0] * 4 for _ in range(4)]
    solutions = []
    def visit():
        best = None
        for r in range(4):
            for c in range(4):
                if board[r][c]:
                    continue
                used = set(board[r]) | {board[i][c] for i in range(4)}
                used |= {board[i][j] for i in range(r//2*2, r//2*2+2)
                         for j in range(c//2*2, c//2*2+2)}
                domain = sorted({1, 2, 3, 4} - used)
                if not domain:
                    return
                if best is None or len(domain) < len(best[2]):
                    best = r, c, domain
        if best is None:
            solutions.append([row[:] for row in board])
            return
        r, c, domain = best
        for value in domain:
            board[r][c] = value
            visit()
            board[r][c] = 0
    visit()
    return solutions


def main():
    out = ROOT / 'experiments' / 'results'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    original = baseline._total_conflicts
    for seed in range(1000):
        initial_boards = []
        for method in ('baseline', 'annealing'):
            calls = 0
            def evaluate(board):
                nonlocal calls
                calls += 1
                return original(board)
            start = time.perf_counter()
            if method == 'baseline':
                baseline._total_conflicts = evaluate
                try:
                    first = None
                    for event in baseline.min_conflicts_steps(rng=random.Random(seed)):
                        if first is None:
                            first = event['board']
                    board = event['board'] if event.get('solved') else None
                    moves = swaps = event['swaps']
                    restart = event['restart']
                finally:
                    baseline._total_conflicts = original
            else:
                board, moves, swaps, restart, first = anneal(seed, evaluate)
            elapsed = (time.perf_counter() - start) * 1000
            if board is not None:
                assert is_valid_solution(board)
            initial_boards.append(first)
            rows.append(dict(seed=seed, method=method, solved=board is not None,
                             proposals=moves, accepted_swaps=swaps, evaluations=calls,
                             restarts=restart, milliseconds=elapsed))
        assert initial_boards[0] == initial_boards[1]
    with (out / 'sudoku_runs.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for method in ('baseline', 'annealing'):
        selected = [r for r in rows if r['method'] == method]
        summary[method] = {'successes': sum(r['solved'] for r in selected)}
        for key in ('proposals', 'accepted_swaps', 'evaluations', 'milliseconds'):
            values = sorted(r[key] for r in selected)
            summary[method][key] = dict(mean=statistics.mean(values), median=statistics.median(values), p95=values[949], maximum=max(values))
        summary[method]['restarted_runs'] = sum(r['restarts'] > 0 for r in selected)
    def naive():
        solutions = []
        backtrack([[0]*4 for _ in range(4)], solutions)
        return solutions
    reference = None
    for name, fn in [('backtracking', naive), ('mrv', enumerate_mrv)]:
        times = []
        for _ in range(30):
            start = time.perf_counter()
            solutions = fn()
            times.append((time.perf_counter() - start)*1000)
            canonical = {tuple(v for row in b for v in row) for b in solutions}
            assert len(canonical) == len(solutions) == 288
            assert all(is_valid_solution(b) for b in solutions)
            if reference is None:
                reference = canonical
            assert reference == canonical
        summary[name] = dict(solutions=288, repeats=30, median_ms=statistics.median(times))
    (out / 'sudoku_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
