"""
search.py – Task 3: Search Algorithms
======================================
Implements BFS, DFS, Greedy Best-First Search, and A* from scratch.
All share a common SearchResult dataclass.

Node-expansion definition (consistent across all algorithms)
------------------------------------------------------------
`nodes_expanded` is incremented each time a node is **popped from the
frontier and its successors are generated**.  Nodes that are discovered
but never popped (e.g., still in frontier at termination) are NOT counted.

Heuristic: Manhattan distance
-------------------------------
  h(state) = |r - r_goal| + |c - c_goal|

Admissibility proof (brief):
  Each step costs at least 1 (penalties only add, never subtract).
  Each move changes the Manhattan distance by exactly 1.
  Therefore h(n) ≤ true remaining cost → admissible.

Consistency proof (brief):
  For any move from n to n' with cost c(n,n'),
    h(n) ≤ c(n,n') + h(n')
  because c(n,n') ≥ 1 and |h(n) - h(n')| ≤ 1 (triangle inequality).
  Consistent ⇒ A* is optimal and never needs to re-expand nodes.
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from problem import PathProblem

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Uniform result object returned by every search algorithm."""
    algorithm:      str
    path:           List[Tuple[int, int]]  # empty if not found
    path_length:    int                    # len(path)-1; -1 if not found
    path_cost:      float                  # penalised cost; 0 if not found
    nodes_expanded: int
    time_ms:        float                  # milliseconds (perf_counter)
    reached_goal:   bool
    explored:       List[Tuple[int, int]]  # expansion order


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def manhattan(state: Tuple[int, int], goal: Tuple[int, int]) -> float:
    """
    Manhattan distance heuristic.
    Admissible and consistent for 4-connected grids with step cost ≥ 1.
    """
    return float(abs(state[0] - goal[0]) + abs(state[1] - goal[1]))


# ---------------------------------------------------------------------------
# Path reconstruction helper
# ---------------------------------------------------------------------------

def _reconstruct_path(
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
    goal:   Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Trace parent pointers from goal back to start and reverse."""
    path = []
    node: Optional[Tuple[int, int]] = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------

def bfs(problem: PathProblem) -> SearchResult:
    """
    Breadth-First Search.

    Data structure : FIFO deque (queue).
    Priority       : none – purely by insertion order (level by level).
    Completeness   : yes (finite graph).
    Optimality     : optimal in number of STEPS (not penalised cost),
                     because it expands nodes level-by-level.
    Ignores        : proximity penalties for path selection; uses only
                     the grid's traversability.

    nodes_expanded is incremented when a node is popped and expanded.
    """
    start  = problem.start
    goal   = problem.goal

    t0 = time.perf_counter()

    frontier:       deque[Tuple[int, int]] = deque([start])
    parent:         Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    nodes_expanded: int = 0
    explored_order: List[Tuple[int, int]] = []

    while frontier:
        node = frontier.popleft()
        nodes_expanded += 1
        explored_order.append(node)

        if problem.is_goal(node):
            path = _reconstruct_path(parent, node)
            t1   = time.perf_counter()
            return SearchResult(
                algorithm      = "BFS",
                path           = path,
                path_length    = len(path) - 1,
                path_cost      = problem.path_cost(path),
                nodes_expanded = nodes_expanded,
                time_ms        = (t1 - t0) * 1000,
                reached_goal   = True,
                explored       = explored_order,
            )

        for (ns, _) in problem.successors(node):
            if ns not in parent:
                parent[ns] = node
                frontier.append(ns)

    t1 = time.perf_counter()
    return SearchResult(
        algorithm      = "BFS",
        path           = [],
        path_length    = -1,
        path_cost      = 0.0,
        nodes_expanded = nodes_expanded,
        time_ms        = (t1 - t0) * 1000,
        reached_goal   = False,
        explored       = explored_order,
    )


# ---------------------------------------------------------------------------
# DFS
# ---------------------------------------------------------------------------

def dfs(problem: PathProblem) -> SearchResult:
    """
    Depth-First Search (graph search with explored set).

    Data structure : LIFO stack (Python list).
    Priority       : none – exploration order determined by action order
                     (UP, RIGHT, DOWN, LEFT as defined in problem.py).
    Completeness   : yes on finite graphs (explored set prevents cycles).
    Optimality     : NOT optimal – returns the first path found,
                     which may be long and winding.

    Neighbour order is fixed (UP, RIGHT, DOWN, LEFT) for reproducibility.
    nodes_expanded is incremented when a node is popped and expanded.
    """
    start  = problem.start
    goal   = problem.goal

    t0 = time.perf_counter()

    # Stack stores (node, parent_node)
    stack:          List[Tuple[Tuple[int,int], Optional[Tuple[int,int]]]] = [(start, None)]
    parent:         Dict[Tuple[int, int], Optional[Tuple[int, int]]]      = {}
    visited:        set                                                     = set()
    nodes_expanded: int                                                     = 0
    explored_order: List[Tuple[int, int]]                                   = []

    while stack:
        node, par = stack.pop()

        if node in visited:
            continue
        visited.add(node)
        parent[node] = par
        nodes_expanded += 1
        explored_order.append(node)

        if problem.is_goal(node):
            path = _reconstruct_path(parent, node)
            t1   = time.perf_counter()
            return SearchResult(
                algorithm      = "DFS",
                path           = path,
                path_length    = len(path) - 1,
                path_cost      = problem.path_cost(path),
                nodes_expanded = nodes_expanded,
                time_ms        = (t1 - t0) * 1000,
                reached_goal   = True,
                explored       = explored_order,
            )

        # Push successors in REVERSE order so UP is processed first
        succs = problem.successors(node)
        for (ns, _) in reversed(succs):
            if ns not in visited:
                stack.append((ns, node))

    t1 = time.perf_counter()
    return SearchResult(
        algorithm      = "DFS",
        path           = [],
        path_length    = -1,
        path_cost      = 0.0,
        nodes_expanded = nodes_expanded,
        time_ms        = (t1 - t0) * 1000,
        reached_goal   = False,
        explored       = explored_order,
    )


# ---------------------------------------------------------------------------
# Greedy Best-First Search
# ---------------------------------------------------------------------------

def greedy_best_first(
    problem:   PathProblem,
    heuristic: Callable[[Tuple[int,int], Tuple[int,int]], float] = manhattan,
) -> SearchResult:
    """
    Greedy Best-First Search.

    Data structure : min-heap priority queue keyed on h(n).
    Priority       : h(n) only (ignores path cost so far).
    Completeness   : yes on finite graphs (with explored set).
    Optimality     : NOT optimal – selects whichever node looks closest
                     to the goal; may walk through penalised zones.
    Tie-break      : insertion counter (deterministic).

    nodes_expanded is incremented when a node is popped and expanded.
    """
    start  = problem.start
    goal   = problem.goal

    t0      = time.perf_counter()
    counter = itertools.count()   # tie-breaking counter

    # heap entry: (h, counter, node)
    heap:           List                                                    = []
    heapq.heappush(heap, (heuristic(start, goal), next(counter), start))

    parent:         Dict[Tuple[int, int], Optional[Tuple[int, int]]]       = {start: None}
    visited:        set                                                      = set()
    nodes_expanded: int                                                      = 0
    explored_order: List[Tuple[int, int]]                                    = []

    while heap:
        h_val, _, node = heapq.heappop(heap)

        if node in visited:
            continue
        visited.add(node)
        nodes_expanded += 1
        explored_order.append(node)

        if problem.is_goal(node):
            path = _reconstruct_path(parent, node)
            t1   = time.perf_counter()
            return SearchResult(
                algorithm      = "Greedy",
                path           = path,
                path_length    = len(path) - 1,
                path_cost      = problem.path_cost(path),
                nodes_expanded = nodes_expanded,
                time_ms        = (t1 - t0) * 1000,
                reached_goal   = True,
                explored       = explored_order,
            )

        for (ns, _) in problem.successors(node):
            if ns not in visited and ns not in parent:
                parent[ns] = node
                heapq.heappush(
                    heap,
                    (heuristic(ns, goal), next(counter), ns),
                )

    t1 = time.perf_counter()
    return SearchResult(
        algorithm      = "Greedy",
        path           = [],
        path_length    = -1,
        path_cost      = 0.0,
        nodes_expanded = nodes_expanded,
        time_ms        = (t1 - t0) * 1000,
        reached_goal   = False,
        explored       = explored_order,
    )


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------

def a_star(
    problem:   PathProblem,
    heuristic: Callable[[Tuple[int,int], Tuple[int,int]], float] = manhattan,
    start:     Optional[Tuple[int, int]] = None,
) -> SearchResult:
    """
    A* Search.

    Data structure : min-heap priority queue keyed on f(n) = g(n) + h(n).
    g(n)           : penalised cost from start to n.
    h(n)           : Manhattan distance heuristic (admissible, consistent).
    Completeness   : yes.
    Optimality     : yes w.r.t. penalised path cost (consistent heuristic
                     guarantees first expansion = optimal cost).

    Re-opening: not needed with a consistent heuristic; we use a
    best-known-g dict and skip stale heap entries.

    The optional `start` parameter allows replanning from a different
    position (e.g., the robot's current location mid-journey) while
    keeping the same goal and cost function.

    nodes_expanded is incremented when a node is popped and expanded
    (only if it matches the best known g-value).
    """
    init_state = start if start is not None else problem.start
    goal       = problem.goal

    t0      = time.perf_counter()
    counter = itertools.count()

    # heap entry: (f, counter, node)
    heap:    List = []
    heapq.heappush(
        heap,
        (heuristic(init_state, goal), next(counter), init_state),
    )

    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {init_state: None}
    g:      Dict[Tuple[int, int], float]                     = {init_state: 0.0}

    nodes_expanded: int                   = 0
    explored_order: List[Tuple[int, int]] = []

    while heap:
        f_val, _, node = heapq.heappop(heap)

        # Skip stale heap entries (we may have inserted a better g later)
        current_g = g.get(node, float("inf"))
        if f_val > current_g + heuristic(node, goal) + 1e-9:
            continue

        nodes_expanded += 1
        explored_order.append(node)

        if problem.is_goal(node):
            path = _reconstruct_path(parent, node)
            t1   = time.perf_counter()
            return SearchResult(
                algorithm      = "A*",
                path           = path,
                path_length    = len(path) - 1,
                path_cost      = g[node],   # exact accumulated cost
                nodes_expanded = nodes_expanded,
                time_ms        = (t1 - t0) * 1000,
                reached_goal   = True,
                explored       = explored_order,
            )

        for (ns, cost) in problem.successors(node):
            new_g = current_g + cost
            if new_g < g.get(ns, float("inf")):
                g[ns]      = new_g
                parent[ns] = node
                f_new      = new_g + heuristic(ns, goal)
                heapq.heappush(heap, (f_new, next(counter), ns))

    t1 = time.perf_counter()
    return SearchResult(
        algorithm      = "A*",
        path           = [],
        path_length    = -1,
        path_cost      = 0.0,
        nodes_expanded = nodes_expanded,
        time_ms        = (t1 - t0) * 1000,
        reached_goal   = False,
        explored       = explored_order,
    )


# ---------------------------------------------------------------------------
# Path validator
# ---------------------------------------------------------------------------

def validate_path(
    path:    List[Tuple[int, int]],
    problem: PathProblem,
) -> bool:
    """
    Assert that every cell in `path` is traversable, the path is
    contiguous (4-connected moves only), and it connects start to goal.

    Returns True if valid, raises AssertionError with details otherwise.
    """
    if not path:
        return True  # empty path allowed when no route exists

    assert path[0] == problem.start, (
        f"Path does not start at start: {path[0]} != {problem.start}"
    )
    assert path[-1] == problem.goal, (
        f"Path does not end at goal: {path[-1]} != {problem.goal}"
    )
    for i, (r, c) in enumerate(path):
        assert problem.is_traversable(r, c), (
            f"Path step {i} at {(r,c)} is NOT traversable "
            f"(cell code {problem.grid[r,c]})"
        )
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        assert dr + dc == 1, (
            f"Non-adjacent step between path[{i}]={path[i]} and "
            f"path[{i+1}]={path[i+1]} (Δr={dr}, Δc={dc})"
        )
    return True

