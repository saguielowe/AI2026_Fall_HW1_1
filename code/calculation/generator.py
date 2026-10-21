"""Deterministic generators and answer checking for the search calculation lab."""

from __future__ import annotations

import hashlib
import math
import random
import secrets
import uuid
from dataclasses import dataclass
from typing import Any


LOCAL_STATE_IDS = list("ABCDEFGHI")
ANNEALING_TEMPERATURES = [8.0, 6.0, 4.0, 3.0, 2.0, 1.0]


def _graph_search_order(
    adjacency: dict[str, list[tuple[str, int]]], strategy: str
) -> list[str]:
    """Run graph search with goal testing when a state is removed from frontier."""
    frontier: list[tuple[str, int]] = [("S", 0)]
    explored: set[str] = set()
    order: list[str] = []
    while frontier:
        if strategy == "dfs":
            state, cost = frontier.pop()
        elif strategy == "bfs":
            state, cost = frontier.pop(0)
        else:
            frontier.sort(key=lambda item: (item[1], item[0]))
            state, cost = frontier.pop(0)
        if state in explored:
            continue
        explored.add(state)
        order.append(state)
        if state == "G":
            return order
        children = adjacency[state]
        if strategy == "dfs":
            children = list(reversed(children))
        for child, weight in children:
            if child not in explored and child not in {item[0] for item in frontier}:
                frontier.append((child, cost + weight))
    return order


def _weighted_graph_search(
    adjacency: dict[str, list[tuple[str, int]]], heuristic: dict[str, int]
) -> tuple[list[str], list[list[dict[str, int | str]]]]:
    """Return A* expansion order and the frontier after each non-goal expansion."""
    frontier: dict[str, int] = {"S": 0}
    explored: set[str] = set()
    order: list[str] = []
    snapshots: list[list[dict[str, int | str]]] = []
    while frontier:
        state = min(frontier, key=lambda item: (frontier[item] + heuristic[item], item))
        cost = frontier.pop(state)
        explored.add(state)
        order.append(state)
        if state == "G":
            break
        for child, weight in adjacency[state]:
            new_cost = cost + weight
            if child in explored:
                continue
            if child not in frontier or new_cost < frontier[child]:
                frontier[child] = new_cost
        snapshots.append(
            [
                {"id": node, "g": g, "f": g + heuristic[node]}
                for node, g in sorted(
                    frontier.items(), key=lambda item: (item[1] + heuristic[item[0]], item[0])
                )
            ]
        )
    return order, snapshots


def _make_frontier_search(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a small graph where A* must replace at least one frontier path."""
    weights = {
        ("S", "A"): rng.randint(7, 9),
        ("S", "B"): rng.randint(1, 2),
        ("S", "C"): rng.randint(4, 6),
        ("B", "A"): rng.randint(1, 2),
        ("B", "D"): rng.randint(7, 9),
        ("A", "D"): rng.randint(1, 2),
        ("A", "E"): rng.randint(5, 7),
        ("C", "B"): rng.randint(4, 6),
        ("C", "E"): rng.randint(5, 7),
        ("D", "G"): rng.randint(2, 3),
        ("E", "G"): rng.randint(3, 5),
    }
    adjacency = {node: [] for node in "SABCDEG"}
    for (source, target), weight in weights.items():
        adjacency[source].append((target, weight))
    for children in adjacency.values():
        children.sort()

    # Exact remaining distances form a consistent heuristic and keep arithmetic friendly.
    heuristic = {"G": 0}
    for node in "EDABCS":
        heuristic[node] = min(weight + heuristic[child] for child, weight in adjacency[node])

    bfs = _graph_search_order(adjacency, "bfs")
    dfs = _graph_search_order(adjacency, "dfs")
    ucs_heuristic = {node: 0 for node in adjacency}
    ucs, _ = _weighted_graph_search(adjacency, ucs_heuristic)
    astar, snapshots = _weighted_graph_search(adjacency, heuristic)
    positions = {
        "S": [55, 190], "A": [215, 65], "B": [215, 190], "C": [215, 315],
        "D": [445, 95], "E": [445, 285], "G": [635, 190],
    }
    public = {
        "nodes": [{"id": node, "h": heuristic[node], "x": positions[node][0], "y": positions[node][1]} for node in adjacency],
        "edges": [{"source": source, "target": target, "cost": cost} for (source, target), cost in weights.items()],
        "snapshot_labels": astar[:-1],
    }
    answer = {"bfs_order": bfs, "dfs_order": dfs, "ucs_order": ucs, "astar_order": astar, "astar_frontiers": snapshots}
    return public, answer


def seed_from_student_id(student_id: str) -> int:
    """Map a student id to a stable 64-bit seed without using Python's hash()."""
    normalized = student_id.strip()
    if not normalized:
        raise ValueError("请输入学号")
    digest = hashlib.sha256(("AI2026-search:" + normalized).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _hill_climbing(scores: list[int], start: int) -> list[int]:
    """Steepest-ascent hill climbing on a one-dimensional neighborhood."""
    route = [start]
    current = start
    while True:
        neighbors = [index for index in (current - 1, current + 1) if 0 <= index < len(scores)]
        best = max(neighbors, key=lambda index: scores[index])
        if scores[best] <= scores[current]:
            return route
        current = best
        route.append(current)


def _annealing_run(
    rng: random.Random, scores: list[int], start: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current = start
    public_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    for step, temperature in enumerate(ANNEALING_TEMPERATURES, 1):
        neighbors = [index for index in (current - 1, current + 1) if 0 <= index < len(scores)]
        candidate = rng.choice(neighbors)
        random_value = round(rng.uniform(0.05, 0.95), 2)
        delta = scores[candidate] - scores[current]
        probability = 1.0 if delta >= 0 else math.exp(delta / temperature)
        accepted = random_value < probability
        next_state = candidate if accepted else current
        public_rows.append(
            {
                "step": step,
                "temperature": temperature,
                "candidate": LOCAL_STATE_IDS[candidate],
                "random": random_value,
            }
        )
        answer_rows.append(
            {
                "current": LOCAL_STATE_IDS[current],
                "delta": delta,
                "exponent": round(delta / temperature, 2),
                "accepted": accepted,
                "next": LOCAL_STATE_IDS[next_state],
            }
        )
        current = next_state
    return public_rows, answer_rows


def _make_local_search(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a similarly sized hill-climbing and simulated-annealing exercise."""
    for _ in range(2000):
        scores = rng.sample(range(1, 22), len(LOCAL_STATE_IDS))
        start = rng.randint(1, len(scores) - 2)
        hill_route = _hill_climbing(scores, start)
        hill_end = hill_route[-1]
        if not (2 <= len(hill_route) <= 4):
            continue
        if scores[hill_end] == max(scores):
            continue

        public_rows, answer_rows = _annealing_run(rng, scores, start)
        worse_rows = [row for row in answer_rows if row["delta"] < 0]
        if len(worse_rows) < 2:
            continue
        if not any(row["accepted"] for row in worse_rows):
            continue
        if not any(not row["accepted"] for row in worse_rows):
            continue
        if not any(row["delta"] >= 0 for row in answer_rows):
            continue
        # The interface displays p to two decimals.  Never generate equality,
        # and keep worse-move comparisons comfortably away from the boundary.
        if any(
            round(
                1.0
                if row["delta"] >= 0
                else math.exp(row["exponent"]),
                2,
            )
            == public_rows[index]["random"]
            or (
                row["delta"] < 0
                and abs(
                    math.exp(row["exponent"])
                    - public_rows[index]["random"]
                )
                < 0.08
            )
            for index, row in enumerate(answer_rows)
        ):
            continue

        visited = [start]
        for row in answer_rows:
            visited.append(LOCAL_STATE_IDS.index(row["next"]))
        best_index = max(visited, key=lambda index: scores[index])
        public = {
            "states": [
                {"id": state_id, "score": scores[index], "index": index}
                for index, state_id in enumerate(LOCAL_STATE_IDS)
            ],
            "state_ids": LOCAL_STATE_IDS,
            "start": LOCAL_STATE_IDS[start],
            "annealing_steps": public_rows,
        }
        answer = {
            "hill_route": [LOCAL_STATE_IDS[index] for index in hill_route],
            "hill_final_score": scores[hill_end],
            "annealing_rows": answer_rows,
            "annealing_final": answer_rows[-1]["next"],
            "annealing_best": LOCAL_STATE_IDS[best_index],
        }
        return public, answer
    raise RuntimeError("无法生成难度合适的局部搜索题目")


def _make_beam_search(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create two rounds of local beam search and expose every possible branch."""
    first_values = rng.sample(range(5, 25), 4)
    first = [{"id": node, "h": value} for node, value in zip("ABCD", first_values)]
    selected_first = sorted(first, key=lambda item: (item["h"], item["id"]))[:2]
    second: dict[str, list[dict[str, Any]]] = {}
    values = iter(rng.sample(range(1, 31), 12))
    for parent in "ABCD":
        second[parent] = [{"id": f"{parent}{index}", "h": next(values)} for index in range(1, 4)]
    candidates = [child for parent in selected_first for child in second[parent["id"]]]
    selected_second = sorted(candidates, key=lambda item: (item["h"], item["id"]))[:2]
    answer = {
        "first": sorted(item["id"] for item in selected_first),
        "explored": sorted(item["id"] for item in candidates),
        "second": sorted(item["id"] for item in selected_second),
        "final": selected_second[0]["id"],
    }
    return {"k": 2, "first": first, "second": second, "explored_count": 6}, answer


def _roulette_pick(ids: list[str], probabilities: list[float], value: float) -> str:
    cumulative = 0.0
    for item_id, probability in zip(ids, probabilities):
        cumulative += probability
        if value < cumulative:
            return item_id
    return ids[-1]


def _make_genetic(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create one fully specified selection/crossover/mutation generation."""
    chromosomes = rng.sample(range(1, 63), 4)
    bits = [f"{value:06b}" for value in chromosomes]
    weights = rng.sample(range(1, 8), 6)
    fitness = [1 + sum(weight * int(bit) for weight, bit in zip(weights, chromosome)) for chromosome in bits]
    total = sum(fitness)
    probabilities = [value / total for value in fitness]
    ids = [f"P{index}" for index in range(1, 5)]
    bounds = [sum(probabilities[:index]) for index in range(1, 5)]
    safe_intervals = []
    low = 0.0
    for high in bounds:
        left, right = low + 0.03, high - 0.03
        if left < right:
            safe_intervals.append((left, right))
        low = high
    first_interval, second_interval = rng.sample(safe_intervals, 2)
    random_values = [round(rng.uniform(*first_interval), 2), round(rng.uniform(*second_interval), 2)]
    parents = [_roulette_pick(ids, probabilities, value) for value in random_values]
    point = rng.randint(2, 4)
    p1, p2 = (bits[ids.index(parent)] for parent in parents)
    crossed = [p1[:point] + p2[point:], p2[:point] + p1[point:]]
    mutation_positions = [rng.randint(1, 6), rng.randint(1, 6)]
    mutated = []
    for child, position in zip(crossed, mutation_positions):
        chars = list(child)
        chars[position - 1] = "1" if chars[position - 1] == "0" else "0"
        mutated.append("".join(chars))
    final_fitness = [1 + sum(weight * int(bit) for weight, bit in zip(weights, child)) for child in mutated]
    public = {
        "individuals": [{"id": item_id, "bits": chromosome} for item_id, chromosome in zip(ids, bits)],
        "weights": weights,
        "random_values": random_values,
        "crossover_point": point,
        "mutation_positions": mutation_positions,
    }
    answer = {
        "fitness": dict(zip(ids, fitness)),
        "probabilities": dict(zip(ids, probabilities)),
        "parents": parents,
        "crossed": crossed,
        "mutated": mutated,
        "final_fitness": final_fitness,
    }
    return public, answer


@dataclass
class GameNode:
    id: str
    depth: int
    children: list["GameNode"]
    utility: int | None = None


def _minimax(node: GameNode, maximizing: bool, values: dict[str, int]) -> int:
    if not node.children:
        assert node.utility is not None
        return node.utility
    child_values = [_minimax(child, not maximizing, values) for child in node.children]
    value = max(child_values) if maximizing else min(child_values)
    values[node.id] = value
    return value


def _alpha_beta(
    node: GameNode,
    maximizing: bool,
    alpha: float,
    beta: float,
    visited: set[str],
) -> int:
    visited.add(node.id)
    if not node.children:
        assert node.utility is not None
        return node.utility
    if maximizing:
        value = -10**9
        for child in node.children:
            value = max(value, _alpha_beta(child, False, alpha, beta, visited))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    value = 10**9
    for child in node.children:
        value = min(value, _alpha_beta(child, True, alpha, beta, visited))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def _flatten(root: GameNode) -> list[GameNode]:
    result: list[GameNode] = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        result.append(node)
        queue.extend(node.children)
    return result


def _make_game_tree(rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    for _ in range(1000):
        counters = {0: 0, 1: 0, 2: 0, 3: 0}

        def build(depth: int) -> GameNode:
            prefix = "R" if depth == 0 else ("N" if depth < 3 else "L")
            counters[depth] += 1
            node_id = "R" if depth == 0 else f"{prefix}{depth}-{counters[depth]}"
            if depth == 3:
                return GameNode(node_id, depth, [], rng.randint(-9, 9))
            count = 3 if depth == 0 else rng.choice([2, 3])
            return GameNode(node_id, depth, [build(depth + 1) for _ in range(count)])

        root = build(0)
        nodes = _flatten(root)
        values: dict[str, int] = {}
        root_value = _minimax(root, True, values)
        visited: set[str] = set()
        alpha_beta_value = _alpha_beta(root, True, -float("inf"), float("inf"), visited)
        pruned = {node.id for node in nodes if node.id not in visited}
        leaves = [node for node in nodes if not node.children]
        evaluated = sum(node.id in visited for node in leaves)
        if not (28 <= len(nodes) <= 34):
            continue
        if alpha_beta_value != root_value or not (3 <= len(pruned) <= 11):
            continue
        if not (10 <= evaluated <= 20):
            continue

        public_nodes = [
            {
                "id": node.id,
                "depth": node.depth,
                "utility": node.utility,
                "children": [child.id for child in node.children],
            }
            for node in nodes
        ]
        best_children = [child.id for child in root.children if values[child.id] == root_value]
        public = {
            "nodes": public_nodes,
            "root": root.id,
            "internal_ids": [node.id for node in nodes if node.children],
            "all_ids": [node.id for node in nodes],
            "root_children": [child.id for child in root.children],
            "leaf_count": len(leaves),
        }
        answer = {
            "values": values,
            "pruned": sorted(pruned),
            "best_children": best_children,
            "evaluated_count": evaluated,
        }
        return public, answer
    raise RuntimeError("无法生成剪枝规模合适的博弈树")


def generate_question(seed: int, source_label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)
    frontier_public, frontier_answer = _make_frontier_search(rng)
    local_public, local_answer = _make_local_search(rng)
    beam_public, beam_answer = _make_beam_search(rng)
    genetic_public, genetic_answer = _make_genetic(rng)
    game_public, game_answer = _make_game_tree(rng)
    question_id = uuid.uuid4().hex
    public = {
        "question_id": question_id,
        "source_label": source_label,
        "frontier_search": frontier_public,
        "local_search": local_public,
        "beam_search": beam_public,
        "genetic": genetic_public,
        "game": game_public,
    }
    answer = {
        "frontier_search": frontier_answer,
        "local_search": local_answer,
        "beam_search": beam_answer,
        "genetic": genetic_answer,
        "game": game_answer,
    }
    return public, answer


def generate_practice_question() -> tuple[dict[str, Any], dict[str, Any]]:
    seed = secrets.randbits(64)
    return generate_question(seed, f"练习题 #{seed % 1000000:06d}")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def check_answer(submission: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    """Return field-level correctness without disclosing the solution."""
    submitted_frontier = submission.get("frontier_search", {})
    correct_frontier = answer["frontier_search"]
    frontier_results = {
        key: submitted_frontier.get(key, []) == correct_frontier[key]
        for key in ("bfs_order", "dfs_order", "ucs_order", "astar_order")
    }
    submitted_snapshots = submitted_frontier.get("astar_frontiers", [])
    snapshot_results = []
    for index, expected in enumerate(correct_frontier["astar_frontiers"]):
        actual = submitted_snapshots[index] if index < len(submitted_snapshots) else []
        normalized = [
            {"id": str(item.get("id", "")).upper(), "g": _as_int(item.get("g")), "f": _as_int(item.get("f"))}
            for item in actual if isinstance(item, dict)
        ]
        snapshot_results.append(normalized == expected)
    frontier_results["astar_frontiers"] = snapshot_results

    submitted_local = submission.get("local_search", {})
    correct_local = answer["local_search"]
    submitted_route = submitted_local.get("hill_route", [])
    local_results = {
        "hill_route": submitted_route == correct_local["hill_route"],
        "hill_final_score": _as_int(submitted_local.get("hill_final_score"))
        == correct_local["hill_final_score"],
    }
    if not local_results["hill_route"]:
        first_wrong = next(
            (
                index + 1
                for index, (actual, expected) in enumerate(
                    zip(submitted_route, correct_local["hill_route"])
                )
                if actual != expected
            ),
            min(len(submitted_route), len(correct_local["hill_route"])) + 1,
        )
        local_results["hill_first_wrong_step"] = first_wrong
    submitted_rows = submitted_local.get("annealing_rows", [])
    row_results = []
    for index, expected in enumerate(correct_local["annealing_rows"]):
        actual = submitted_rows[index] if index < len(submitted_rows) else {}
        exponent = _as_float(actual.get("exponent"))
        row_results.append(
            {
                "delta": _as_int(actual.get("delta")) == expected["delta"],
                "exponent": exponent is not None
                and abs(exponent - expected["exponent"]) <= 0.005,
                "accepted": actual.get("accepted") is expected["accepted"],
                "next": actual.get("next") == expected["next"],
            }
        )
    local_results.update(
        {
            "annealing_rows": row_results,
            "annealing_final": submitted_local.get("annealing_final")
            == correct_local["annealing_final"],
            "annealing_best": submitted_local.get("annealing_best")
            == correct_local["annealing_best"],
        }
    )

    submitted_beam = submission.get("beam_search", {})
    correct_beam = answer["beam_search"]
    beam_results = {
        "first": sorted(submitted_beam.get("first", [])) == correct_beam["first"],
        "explored": sorted(submitted_beam.get("explored", []))
        == correct_beam["explored"],
        "second": sorted(submitted_beam.get("second", [])) == correct_beam["second"],
        "final": submitted_beam.get("final") == correct_beam["final"],
    }

    submitted_genetic = submission.get("genetic", {})
    correct_genetic = answer["genetic"]
    total_fitness = sum(correct_genetic["fitness"].values())

    def probability_is_correct(item_id: str, expected: float) -> bool:
        actual = submitted_genetic.get("probabilities", {}).get(item_id)
        if isinstance(actual, dict):
            return (
                _as_int(actual.get("numerator")) == correct_genetic["fitness"][item_id]
                and _as_int(actual.get("denominator")) == total_fitness
            )
        # 保留对旧版数字提交格式的兼容。
        value = _as_float(actual)
        return value is not None and abs(value - expected) <= 0.0051

    genetic_results = {
        "fitness": {
            item_id: _as_int(submitted_genetic.get("fitness", {}).get(item_id)) == value
            for item_id, value in correct_genetic["fitness"].items()
        },
        "probabilities": {
            item_id: probability_is_correct(item_id, expected)
            for item_id, expected in correct_genetic["probabilities"].items()
        },
        "parents": submitted_genetic.get("parents", []) == correct_genetic["parents"],
        "crossed": submitted_genetic.get("crossed", []) == correct_genetic["crossed"],
        "mutated": submitted_genetic.get("mutated", []) == correct_genetic["mutated"],
        "final_fitness": [
            _as_int(value) for value in submitted_genetic.get("final_fitness", [])
        ] == correct_genetic["final_fitness"],
    }

    submitted_game = submission.get("game", {})
    submitted_values = submitted_game.get("values", {})
    value_results = {
        node_id: _as_int(submitted_values.get(node_id)) == value
        for node_id, value in answer["game"]["values"].items()
    }
    game_results = {
        "values": value_results,
        "pruned": sorted(submitted_game.get("pruned", [])) == answer["game"]["pruned"],
        "best_child": submitted_game.get("best_child") in answer["game"]["best_children"],
        "evaluated_count": _as_int(submitted_game.get("evaluated_count")) == answer["game"]["evaluated_count"],
    }
    submitted_pruned = set(submitted_game.get("pruned", []))
    correct_pruned = set(answer["game"]["pruned"])
    game_results["pruned_missing_count"] = len(correct_pruned - submitted_pruned)
    game_results["pruned_extra_count"] = len(submitted_pruned - correct_pruned)

    graded_flags = [
        *(frontier_results[key] for key in ("bfs_order", "dfs_order", "ucs_order", "astar_order")),
        *snapshot_results,
        local_results["hill_route"], local_results["hill_final_score"],
        *(flag for row in row_results for flag in row.values()),
        local_results["annealing_final"], local_results["annealing_best"],
        *beam_results.values(),
        *genetic_results["fitness"].values(), *genetic_results["probabilities"].values(),
        genetic_results["parents"], genetic_results["crossed"], genetic_results["mutated"],
        genetic_results["final_fitness"],
    ]
    practice_flags = [
        *value_results.values(),
        game_results["pruned"],
        game_results["best_child"],
        game_results["evaluated_count"],
    ]
    practice_attempted = (
        any(str(value).strip() for value in submitted_values.values())
        or bool(submitted_game.get("pruned", []))
        or str(submitted_game.get("best_child", "")).strip() != ""
        or str(submitted_game.get("evaluated_count", "")).strip() != ""
    )
    graded_summary = {
        "all_correct": all(graded_flags),
        "correct_count": sum(graded_flags),
        "total_count": len(graded_flags),
    }
    practice_summary = {
        "attempted": practice_attempted,
        "all_correct": all(practice_flags),
        "correct_count": sum(practice_flags),
        "total_count": len(practice_flags),
    }
    return {
        # 保留顶层字段以兼容原有测试和调用方；它们始终只表示计分部分。
        "all_correct": graded_summary["all_correct"],
        "correct_count": graded_summary["correct_count"],
        "total_count": graded_summary["total_count"],
        "graded": graded_summary,
        "practice": practice_summary,
        "frontier_search": frontier_results,
        "local_search": local_results,
        "beam_search": beam_results,
        "genetic": genetic_results,
        "game": game_results,
    }
