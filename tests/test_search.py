"""
tests/test_search.py
====================
Unit tests for potato-field robot path planning:
  - Path validity (all steps traversable, 4-connected, start-to-goal)
  - BFS optimality in steps (fewest steps)
  - A* optimality in penalised cost (cheapest cost)
  - Goal unreachable handling (returns reached_goal=False, empty path)
  - Crop-safe path differentiation (with vs without proximity penalty)
  - Online replanning triggering and goal attainment
"""

import pytest
import numpy as np

from field import Field, Cell, SAFE, POTATO_ROW, PROTECTED, OBSTACLE, build_field
from problem import PathProblem
from search import (
    bfs,
    dfs,
    greedy_best_first,
    a_star,
    manhattan,
    validate_path,
    SearchResult,
)
from replanning import run_replanning


@pytest.fixture
def standard_field():
    """Return the standard 15x15 test field."""
    return build_field()


def test_path_validity_all_algorithms(standard_field):
    """Ensure all four algorithms produce valid, legal paths."""
    prob = PathProblem(standard_field.known_map, standard_field.start, standard_field.goal, use_penalty=True)

    results = [
        bfs(prob),
        dfs(prob),
        greedy_best_first(prob, manhattan),
        a_star(prob, manhattan),
    ]

    for res in results:
        assert res.reached_goal is True, f"{res.algorithm} failed to reach goal"
        assert len(res.path) > 0
        assert validate_path(res.path, prob) is True


def test_bfs_optimality_in_steps(standard_field):
    """Verify BFS finds the minimum number of steps among all algorithms."""
    prob = PathProblem(standard_field.known_map, standard_field.start, standard_field.goal, use_penalty=True)

    res_bfs = bfs(prob)
    res_dfs = dfs(prob)
    res_greedy = greedy_best_first(prob, manhattan)
    res_astar = a_star(prob, manhattan)

    assert res_bfs.reached_goal is True
    assert res_bfs.path_length <= res_dfs.path_length
    assert res_bfs.path_length <= res_greedy.path_length
    assert res_bfs.path_length <= res_astar.path_length


def test_astar_optimality_in_penalised_cost(standard_field):
    """Verify A* finds the minimum penalised path cost."""
    prob = PathProblem(standard_field.known_map, standard_field.start, standard_field.goal, use_penalty=True)

    res_bfs = bfs(prob)
    res_dfs = dfs(prob)
    res_greedy = greedy_best_first(prob, manhattan)
    res_astar = a_star(prob, manhattan)

    assert res_astar.reached_goal is True
    assert res_astar.path_cost <= res_bfs.path_cost + 1e-6
    assert res_astar.path_cost <= res_dfs.path_cost + 1e-6
    assert res_astar.path_cost <= res_greedy.path_cost + 1e-6


def test_goal_unreachable_handling():
    """Verify graceful handling when the goal is completely blocked."""
    field_obj = build_field()
    blocked_grid = field_obj.known_map.copy()

    # Block both incoming paths to goal (14, 14): (13, 14) and (14, 13)
    blocked_grid[13, 14] = OBSTACLE
    blocked_grid[14, 13] = OBSTACLE

    prob = PathProblem(blocked_grid, field_obj.start, field_obj.goal, use_penalty=True)

    for search_fn in [
        lambda: bfs(prob),
        lambda: dfs(prob),
        lambda: greedy_best_first(prob, manhattan),
        lambda: a_star(prob, manhattan),
    ]:
        res = search_fn()
        assert res.reached_goal is False
        assert res.path == []
        assert res.path_length == -1
        assert res.path_cost == 0.0


def test_crop_safe_difference(standard_field):
    """Verify that proximity penalty changes the planned path away from protected zones."""
    p_no = PathProblem(standard_field.known_map, standard_field.start, standard_field.goal, use_penalty=False)
    p_yes = PathProblem(standard_field.known_map, standard_field.start, standard_field.goal, use_penalty=True)

    r_no = a_star(p_no, manhattan)
    r_yes = a_star(p_yes, manhattan)

    assert r_no.reached_goal is True
    assert r_yes.reached_goal is True
    # The paths must differ
    assert r_no.path != r_yes.path

    # The penalised cost of the safe path must be lower than the unpenalised path evaluated under penalties
    cost_unpenalised_under_penalties = p_yes.path_cost(r_no.path)
    cost_penalised_path = r_yes.path_cost
    assert cost_penalised_path < cost_unpenalised_under_penalties


def test_replanning_triggers_and_succeeds(standard_field):
    """Verify online replanning triggers at least 1 replan and still reaches the goal."""
    res = run_replanning(
        true_map=standard_field.true_map,
        known_map_in=standard_field.known_map,
        start=standard_field.start,
        goal=standard_field.goal,
        hidden_obs=standard_field.hidden_obstacles,
        sensor_range=2,
    )

    assert res.reached_goal is True
    assert res.replan_count >= 1
    assert len(res.trail) > 1
    assert res.trail[0] == standard_field.start
    assert res.trail[-1] == standard_field.goal


def test_robot_never_steps_on_obstacle(standard_field):
    """Ensure the robot never enters an obstacle on true_map during replanning."""
    res = run_replanning(
        true_map=standard_field.true_map,
        known_map_in=standard_field.known_map,
        start=standard_field.start,
        goal=standard_field.goal,
        hidden_obs=standard_field.hidden_obstacles,
        sensor_range=2,
    )

    for step, cell in enumerate(res.trail):
        assert standard_field.true_map[cell] == SAFE, (
            f"Robot stepped on non-SAFE cell at step {step}: {cell} "
            f"(code {standard_field.true_map[cell]})"
        )

