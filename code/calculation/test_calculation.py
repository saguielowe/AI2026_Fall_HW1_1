import copy
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator import check_answer, generate_question, seed_from_student_id


def _perfect_submission(answer):
    return copy.deepcopy({
        "frontier_search": answer["frontier_search"],
        "local_search": {
            "hill_route": answer["local_search"]["hill_route"],
            "hill_final_score": answer["local_search"]["hill_final_score"],
            "annealing_rows": answer["local_search"]["annealing_rows"],
            "annealing_final": answer["local_search"]["annealing_final"],
            "annealing_best": answer["local_search"]["annealing_best"],
        },
        "beam_search": answer["beam_search"],
        "genetic": answer["genetic"],
        "game": {
            "values": answer["game"]["values"],
            "pruned": answer["game"]["pruned"],
            "best_child": answer["game"]["best_children"][0],
            "evaluated_count": answer["game"]["evaluated_count"],
        },
    })


def test_student_id_is_stable():
    seed = seed_from_student_id("2026012345")
    assert seed == seed_from_student_id(" 2026012345 ")
    public1, answer1 = generate_question(seed, "test")
    public2, answer2 = generate_question(seed, "test")
    public1.pop("question_id")
    public2.pop("question_id")
    assert public1 == public2
    assert answer1 == answer2


def test_generated_questions_stay_in_target_difficulty_band():
    for seed in range(40):
        public, answer = generate_question(seed, "test")
        assert len(public["frontier_search"]["nodes"]) == 7
        assert answer["frontier_search"]["astar_order"][-1] == "G"
        assert len(answer["frontier_search"]["astar_frontiers"]) == len(answer["frontier_search"]["astar_order"]) - 1
        assert len(public["local_search"]["states"]) == 9
        assert len(public["local_search"]["annealing_steps"]) == 6
        assert 2 <= len(answer["local_search"]["hill_route"]) <= 4
        worse = [row for row in answer["local_search"]["annealing_rows"] if row["delta"] < 0]
        assert any(row["accepted"] for row in worse)
        assert any(not row["accepted"] for row in worse)
        for shown, row in zip(
            public["local_search"]["annealing_steps"],
            answer["local_search"]["annealing_rows"],
        ):
            probability = 1.0 if row["delta"] >= 0 else math.exp(row["exponent"])
            assert round(probability, 2) != shown["random"]
        assert 28 <= len(public["game"]["nodes"]) <= 34
        assert 3 <= len(answer["game"]["pruned"]) <= 11
        assert 10 <= answer["game"]["evaluated_count"] <= 20
        assert len(public["game"]["root_children"]) == 3
        assert len(answer["beam_search"]["first"]) == 2
        assert len(answer["beam_search"]["explored"]) == 6
        assert len(answer["beam_search"]["second"]) == 2
        assert sum(len(children) for children in public["beam_search"]["second"].values()) == 12
        assert len(public["genetic"]["individuals"]) == 4


def test_perfect_answer_is_accepted_and_wrong_answer_is_not():
    _, answer = generate_question(2026, "test")
    submission = _perfect_submission(answer)
    result = check_answer(submission, answer)
    assert result["all_correct"]
    assert result["correct_count"] == result["total_count"]

    submission["local_search"]["hill_final_score"] += 1
    result = check_answer(submission, answer)
    assert not result["all_correct"]
    assert not result["local_search"]["hill_final_score"]


def test_genetic_probabilities_accept_fitness_over_total_fraction():
    _, answer = generate_question(2026, "test")
    submission = _perfect_submission(answer)
    total = sum(answer["genetic"]["fitness"].values())
    submission["genetic"]["probabilities"] = {
        item_id: {"numerator": fitness, "denominator": total}
        for item_id, fitness in answer["genetic"]["fitness"].items()
    }
    result = check_answer(submission, answer)
    assert all(result["genetic"]["probabilities"].values())

    submission["genetic"]["probabilities"]["P1"]["denominator"] = total + 1
    result = check_answer(submission, answer)
    assert not result["genetic"]["probabilities"]["P1"]


def test_feedback_identifies_the_exact_annealing_field_and_pruning_difference():
    _, answer = generate_question(99, "test")
    submission = _perfect_submission(answer)
    submission["local_search"]["annealing_rows"][2]["exponent"] += 0.1
    submission["game"]["pruned"].pop()

    result = check_answer(submission, answer)
    row = result["local_search"]["annealing_rows"][2]
    assert row == {"delta": True, "exponent": False, "accepted": True, "next": True}
    assert result["game"]["pruned_missing_count"] == 1
    assert result["game"]["pruned_extra_count"] == 0
    assert "pruned_missing_ids" not in result["game"]
    assert "pruned_extra_ids" not in result["game"]


def test_two_decimal_exponent_is_accepted():
    answer = {
        "local_search": {
            "hill_route": ["A"],
            "hill_final_score": 1,
            "annealing_rows": [
                {"current": "A", "delta": 4, "exponent": 1.33, "accepted": True, "next": "B"}
            ],
            "annealing_final": "B",
            "annealing_best": "B",
        },
        "frontier_search": {"bfs_order": [], "dfs_order": [], "ucs_order": [], "astar_order": [], "astar_frontiers": []},
        "beam_search": {"first": [], "explored": [], "second": [], "final": ""},
        "genetic": {"fitness": {}, "probabilities": {}, "parents": [], "crossed": [], "mutated": [], "final_fitness": []},
        "game": {"values": {}, "pruned": [], "best_children": ["R"], "evaluated_count": 0},
    }
    submission = {
        "local_search": {
            "hill_route": ["A"],
            "hill_final_score": 1,
            "annealing_rows": [
                {"delta": 4, "exponent": 1.33, "accepted": True, "next": "B"}
            ],
            "annealing_final": "B",
            "annealing_best": "B",
        },
        "frontier_search": {"bfs_order": [], "dfs_order": [], "ucs_order": [], "astar_order": [], "astar_frontiers": []},
        "beam_search": {"first": [], "explored": [], "second": [], "final": ""},
        "genetic": {"fitness": {}, "probabilities": {}, "parents": [], "crossed": [], "mutated": [], "final_fitness": []},
        "game": {"values": {}, "pruned": [], "best_child": "R", "evaluated_count": 0},
    }
    assert check_answer(submission, answer)["all_correct"]


def test_optional_minimax_does_not_affect_grade():
    _, answer = generate_question(7, "test")
    submission = _perfect_submission(answer)
    submission["game"] = {"values": {}, "pruned": [], "best_child": "", "evaluated_count": ""}
    result = check_answer(submission, answer)
    assert result["all_correct"]
    assert result["graded"]["all_correct"]
    assert not result["practice"]["attempted"]


def test_graded_and_optional_practice_results_are_reported_separately():
    _, answer = generate_question(8, "test")
    submission = _perfect_submission(answer)
    submission["game"]["values"]["R"] += 1

    result = check_answer(submission, answer)
    assert result["graded"]["all_correct"]
    assert result["all_correct"]
    assert result["practice"]["attempted"]
    assert not result["practice"]["all_correct"]
    assert result["practice"]["correct_count"] == result["practice"]["total_count"] - 1
