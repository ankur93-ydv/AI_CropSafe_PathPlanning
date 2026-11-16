"""
problem.py – Task 2 & 4: Problem Formulation and Crop-Safe Planning
====================================================================
Formally models the robot path-planning problem and computes the
proximity-penalty map that makes the robot prefer routes away from
protected crop regions.

Problem formulation
-------------------
State       : (row, col) – integer tuple for the robot's grid position.
Initial state: start cell supplied at construction time.
Actions     : UP (-1,0), DOWN (+1,0), LEFT (0,-1), RIGHT (0,+1).
              **4-connected only – diagonal moves are NOT allowed.**
Transition  : result(state, action) → neighbour if in-bounds and SAFE,
              else action is inapplicable.
Goal test   : state == goal.
Path cost   : sum of step_cost(next_state) where
              step_cost(s) = 1 + penalty_map[s].
              Base cost is 1 per step; penalties only add to it.

Penalty map (soft constraint)
-----------------------------
Computed once per map (NumPy float32 array, same shape as the grid).
Constants (configurable at the top of this file):

  PENALTY_D1 = 5   – safe cell at Manhattan distance 1 from PROTECTED
  PENALTY_D2 = 2   – safe cell at Manhattan distance 2 from PROTECTED
  PENALTY_D0 = 0   – all other safe cells

Multiple protected cells may contribute; the MAXIMUM applicable penalty
is used (not the sum), so a cell adjacent to two protected cells still
gets +5, not +10.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from field import Cell, SAFE, PROTECTED, OBSTACLE, POTATO_ROW

# ---------------------------------------------------------------------------
# Configurable penalty constants
# ---------------------------------------------------------------------------

PENALTY_D1: float = 5.0   # adjacent (Manhattan dist 1) to PROTECTED
PENALTY_D2: float = 2.0   # Manhattan dist 2 from PROTECTED
PENALTY_D0: float = 0.0   # otherwise

# ---------------------------------------------------------------------------
# Movement directions (documented order: UP, RIGHT, DOWN, LEFT)
# Used by DFS for reproducibility; all algorithms expand in this order.
# ---------------------------------------------------------------------------

ACTIONS: List[Tuple[int, int]] = [
    (-1,  0),   # UP
    ( 0,  1),   # RIGHT
    ( 1,  0),   # DOWN
    ( 0, -1),   # LEFT
]

ACTION_NAMES: Dict[Tuple[int, int], str] = {
    (-1,  0): "UP",
    ( 0,  1): "RIGHT",
    ( 1,  0): "DOWN",
    ( 0, -1): "LEFT",
}

# ---------------------------------------------------------------------------
# Penalty map builder
# ---------------------------------------------------------------------------

def build_penalty_map(grid: np.ndarray) -> np.ndarray:
    """
    Compute and return the proximity-penalty map for `grid`.

    Rules
    -----
    * Only SAFE cells receive a non-zero penalty (non-SAFE cells are
      never entered, so their penalty is irrelevant, but we set it 0).
    * For each PROTECTED cell, every SAFE neighbour at Manhattan
      distance 1 receives PENALTY_D1 and at distance 2 receives
      PENALTY_D2.  The MAXIMUM over all protected cells is taken.

    Parameters
    ----------
    grid : Integer NumPy array (Cell codes).

    Returns
    -------
    penalty_map : float32 NumPy array, same shape as `grid`.
    """
    rows, cols  = grid.shape
    penalty_map = np.zeros((rows, cols), dtype=np.float32)

    protected_cells = list(zip(*np.where(grid == PROTECTED)))

    for (pr, pc) in protected_cells:
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != SAFE:
                    continue
                dist = abs(r - pr) + abs(c - pc)
                if dist == 1:
                    penalty_map[r, c] = max(penalty_map[r, c], PENALTY_D1)
                elif dist == 2:
                    penalty_map[r, c] = max(penalty_map[r, c], PENALTY_D2)

    return penalty_map


# ---------------------------------------------------------------------------
# PathProblem
# ---------------------------------------------------------------------------

class PathProblem:
    """
    Formal search problem for robot path planning on a 2D grid.

    Parameters
    ----------
    grid        : Integer NumPy array (Cell codes).  The problem
                  operates on whatever map is passed here (true_map,
                  known_map, or an updated known_map for replanning).
    start       : (row, col) initial state.
    goal        : (row, col) goal state.
    use_penalty : If True (default), step costs include the proximity
                  penalty.  Set False for the "without penalty" A* run.
    """

    def __init__(
        self,
        grid:        np.ndarray,
        start:       Tuple[int, int],
        goal:        Tuple[int, int],
        use_penalty: bool = True,
    ) -> None:
        self.grid        = grid
        self.start       = start
        self.goal        = goal
        self.use_penalty = use_penalty
        self.rows, self.cols = grid.shape
        self.penalty_map = (
            build_penalty_map(grid) if use_penalty
            else np.zeros(grid.shape, dtype=np.float32)
        )

    # ------------------------------------------------------------------
    # Core problem interface
    # ------------------------------------------------------------------

    def in_bounds(self, r: int, c: int) -> bool:
        """True if (r, c) is inside the grid."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_traversable(self, r: int, c: int) -> bool:
        """True if (r, c) is in-bounds and is a SAFE cell."""
        return self.in_bounds(r, c) and self.grid[r, c] == SAFE

    def actions(self, state: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Return the list of applicable action tuples (dr, dc) from `state`.
        An action is applicable iff the resulting cell is traversable.
        """
        r, c = state
        return [
            (dr, dc)
            for (dr, dc) in ACTIONS
            if self.is_traversable(r + dr, c + dc)
        ]

    def result(
        self,
        state:  Tuple[int, int],
        action: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Transition model: apply `action` to `state`.
        Assumes the action is legal (use `actions` to filter first).
        """
        return (state[0] + action[0], state[1] + action[1])

    def is_goal(self, state: Tuple[int, int]) -> bool:
        """Goal test: True iff state matches the goal position."""
        return state == self.goal

    def step_cost(self, next_state: Tuple[int, int]) -> float:
        """
        Cost to move INTO `next_state`.
        step_cost = 1 + penalty_map[next_state]
        """
        r, c = next_state
        return 1.0 + float(self.penalty_map[r, c])

    def successors(
        self,
        state: Tuple[int, int],
    ) -> List[Tuple[Tuple[int, int], float]]:
        """
        Return list of (next_state, step_cost) for every legal action
        from `state`.
        """
        result = []
        for action in self.actions(state):
            ns = self.result(state, action)
            result.append((ns, self.step_cost(ns)))
        return result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def path_cost(self, path: List[Tuple[int, int]]) -> float:
        """Compute total penalised cost of a given path list."""
        if len(path) < 2:
            return 0.0
        return sum(self.step_cost(s) for s in path[1:])

    def summary(self) -> str:
        """Return a human-readable problem formulation summary."""
        lines = [
            "=" * 60,
            "PROBLEM FORMULATION",
            "=" * 60,
            f"  Grid size       : {self.rows} × {self.cols}",
            f"  Initial state   : {self.start}",
            f"  Goal state      : {self.goal}",
            "  Actions         : UP, RIGHT, DOWN, LEFT (4-connected, NO diagonals)",
            "  Goal test       : state == goal",
            "  Step cost       : 1 + penalty_map[cell]",
            f"  Use penalty     : {self.use_penalty}",
            f"  Penalty D1 (+{PENALTY_D1}): safe cells at Manhattan dist 1 from PROTECTED",
            f"  Penalty D2 (+{PENALTY_D2}): safe cells at Manhattan dist 2 from PROTECTED",
            "=" * 60,
        ]
        return "\n".join(lines)

