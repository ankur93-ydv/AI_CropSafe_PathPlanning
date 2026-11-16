"""
field.py – Task 1: Field Environment
=====================================
Defines the 15×15 potato-field grid, cell codes, the Field dataclass,
visualisation helpers, and layout validation.

Cell codes
----------
0  SAFE        – traversable soil aisle / headland
1  POTATO_ROW  – planted potato row  (impassable)
2  PROTECTED   – crop-protection region (impassable)
3  OBSTACLE    – stone / tool (impassable)

Hidden obstacles are present in `true_map` only; `known_map` starts
without them and is updated at runtime by the replanning module.
"""

from __future__ import annotations

import sys
from collections import deque
from enum import IntEnum
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Cell codes
# ---------------------------------------------------------------------------

class Cell(IntEnum):
    """Integer codes for each cell type in the grid."""
    SAFE        = 0
    POTATO_ROW  = 1
    PROTECTED   = 2
    OBSTACLE    = 3


# Shorter aliases used in map construction
SAFE       = Cell.SAFE
POTATO_ROW = Cell.POTATO_ROW
PROTECTED  = Cell.PROTECTED
OBSTACLE   = Cell.OBSTACLE

# ---------------------------------------------------------------------------
# Field class
# ---------------------------------------------------------------------------

class Field:
    """
    Encapsulates the potato-field grid.

    Attributes
    ----------
    true_map : np.ndarray
        Ground-truth 15×15 integer grid including hidden obstacles.
    known_map : np.ndarray
        The robot's initial belief map (hidden obstacles are absent).
    start : tuple[int, int]
        Starting cell (row, col) – must be SAFE.
    goal : tuple[int, int]
        Goal cell (row, col) – must be SAFE.
    hidden_obstacles : list[tuple[int, int]]
        Cells that exist in true_map but not in known_map initially.
    """

    def __init__(
        self,
        true_map: np.ndarray,
        known_map: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        hidden_obstacles: List[Tuple[int, int]],
    ) -> None:
        self.true_map         = true_map
        self.known_map        = known_map
        self.start            = start
        self.goal             = goal
        self.hidden_obstacles = hidden_obstacles

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def in_bounds(self, r: int, c: int) -> bool:
        """Return True if (r, c) is inside the grid."""
        rows, cols = self.true_map.shape
        return 0 <= r < rows and 0 <= c < cols

    def is_traversable(self, grid: np.ndarray, r: int, c: int) -> bool:
        """Return True iff (r, c) is in-bounds and is a SAFE cell."""
        return self.in_bounds(r, c) and grid[r, c] == SAFE

    def copy_known_map(self) -> np.ndarray:
        """Return a fresh deep copy of the current known_map."""
        return self.known_map.copy()


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def build_field() -> Field:
    """
    Construct and return the standard 15×15 potato-field.

    Layout (ASCII preview):
      . = SAFE, # = POTATO_ROW, P = PROTECTED, X = known OBSTACLE,
      h = hidden obstacle (true map only), S = start, G = goal

         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
     0   S  .  .  .  .  .  .  .  .  .  .  .  .  .  .
     1   .  #  #  .  #  #  .  #  #  .  #  #  .  .  .
     2   .  #  #  .  #  #  .  #  #  .  #  #  .  P  P
     3   .  #  #  .  #  #  .  #  #  .  #  #  .  P  P
     4   .  #  #  .  #  #  .  #  #  .  #  #  .  .  .
     5   .  #  #  .  #  #  X  #  #  .  #  #  .  .  .
     6   .  #  #  .  #  #  .  #  #  .  #  #  .  .  .
     7   .  .  .  .  .  .  h  .  .  .  .  .  .  .  .
     8   .  #  #  .  #  #  .  #  #  .  #  #  .  .  .
     9   .  #  #  .  #  #  .  #  #  X  #  #  .  P  P
    10   .  #  #  .  #  #  .  #  #  .  #  #  .  P  P
    11   .  #  #  .  #  #  .  #  #  .  #  #  .  .  .
    12   .  #  #  X  #  #  .  #  #  .  #  #  .  .  .
    13   .  #  #  .  #  #  .  #  #  .  #  #  .  .  .
    14   .  .  .  .  .  .  .  .  .  .  .  .  .  .  G

    Hidden obstacles:
      (7, 6)  – lies directly on the optimal A* path (aisle at row 7)
      (3, 3)  – secondary hidden obstacle in the left vertical aisle

    Known obstacles (visible from the start):
      (5, 6), (9, 9), (12, 3)

    Protected regions (impassable):
      rows 2-3, cols 13-14  (top-right block)
      rows 9-10, cols 13-14 (bottom-right block)
    """

    ROWS, COLS = 15, 15
    grid = np.zeros((ROWS, COLS), dtype=np.int32)

    # ------------------------------------------------------------------
    # Potato rows: vertical stripes at columns 1-2, 4-5, 7-8, 10-11
    # Rows 1-6 and 8-13 (leaving headlands at rows 0, 7, 14)
    # ------------------------------------------------------------------
    potato_col_pairs = [(1, 2), (4, 5), (7, 8), (10, 11)]
    crop_rows_range  = list(range(1, 7)) + list(range(8, 14))

    for r in crop_rows_range:
        for (c1, c2) in potato_col_pairs:
            grid[r, c1] = POTATO_ROW
            grid[r, c2] = POTATO_ROW

    # ------------------------------------------------------------------
    # Protected regions (2×2 blocks, adjacent to aisles)
    # ------------------------------------------------------------------
    for r in (2, 3):
        for c in (13, 14):
            grid[r, c] = PROTECTED

    for r in (9, 10):
        for c in (13, 14):
            grid[r, c] = PROTECTED

    # ------------------------------------------------------------------
    # Known obstacles (3 visible from the start)
    # ------------------------------------------------------------------
    known_obstacles = [(5, 6), (9, 9), (12, 3)]
    for (r, c) in known_obstacles:
        grid[r, c] = OBSTACLE

    # ------------------------------------------------------------------
    # Start and goal
    # ------------------------------------------------------------------
    start = (0, 0)
    goal  = (14, 14)

    # Ensure start/goal are SAFE (they should be, but be explicit)
    grid[start] = SAFE
    grid[goal]  = SAFE

    # ------------------------------------------------------------------
    # known_map = copy of grid WITHOUT hidden obstacles
    # ------------------------------------------------------------------
    known_map = grid.copy()

    # ------------------------------------------------------------------
    # Hidden obstacles (added only to true_map)
    # ------------------------------------------------------------------
    hidden_obstacles = [(7, 6), (3, 3)]
    true_map = known_map.copy()
    for (r, c) in hidden_obstacles:
        true_map[r, c] = OBSTACLE

    field = Field(
        true_map         = true_map,
        known_map        = known_map,
        start            = start,
        goal             = goal,
        hidden_obstacles = hidden_obstacles,
    )

    # Validate layout
    validate_layout(field)
    return field


# ---------------------------------------------------------------------------
# Layout validation
# ---------------------------------------------------------------------------

def validate_layout(field: Field) -> None:
    """
    Verify that a path exists from start to goal even after all hidden
    obstacles are revealed (i.e., on the true_map).

    Uses BFS. Raises RuntimeError with a clear message if unreachable.
    Also validates that start and goal are SAFE cells.
    """
    start, goal = field.start, field.goal

    # Start / goal must be SAFE on both maps
    for name, pos, grid in [
        ("start (known_map)", start, field.known_map),
        ("goal (known_map)",  goal,  field.known_map),
        ("start (true_map)",  start, field.true_map),
        ("goal (true_map)",   goal,  field.true_map),
    ]:
        r, c = pos
        if grid[r, c] != SAFE:
            raise RuntimeError(
                f"Layout error: {name} at {pos} is not a SAFE cell "
                f"(value={grid[r, c]})."
            )

    # BFS on true_map to verify reachability
    queue   = deque([start])
    visited = {start}
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return  # reachable
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (
                field.in_bounds(nr, nc)
                and (nr, nc) not in visited
                and field.true_map[nr, nc] == SAFE
            ):
                visited.add((nr, nc))
                queue.append((nr, nc))

    raise RuntimeError(
        f"Layout error: goal {goal} is NOT reachable from start {start} "
        "on true_map. Adjust the hidden obstacles or add an alternate route."
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

# Colour map: index matches Cell int value
_CMAP_COLORS = ["#F5DEB3", "#228B22", "#FF4500", "#2F2F2F"]  # beige, green, red-orange, dark-grey
_CMAP        = matplotlib.colors.ListedColormap(_CMAP_COLORS)
_NORM        = matplotlib.colors.BoundaryNorm([0, 1, 2, 3, 4], _CMAP.N)

_LEGEND_PATCHES = [
    mpatches.Patch(facecolor=_CMAP_COLORS[0], edgecolor="grey",  label="SAFE (aisle / headland)"),
    mpatches.Patch(facecolor=_CMAP_COLORS[1], edgecolor="grey",  label="POTATO_ROW (impassable)"),
    mpatches.Patch(facecolor=_CMAP_COLORS[2], edgecolor="grey",  label="PROTECTED (impassable)"),
    mpatches.Patch(facecolor=_CMAP_COLORS[3], edgecolor="grey",  label="OBSTACLE (impassable)"),
]

# Path and marker colours
_PATH_COLOR   = "#1E90FF"   # dodger blue
_START_COLOR  = "#1E90FF"   # blue
_GOAL_COLOR   = "#FFD700"   # gold
_HIDDEN_COLOR = "#FF69B4"   # hot pink (outline only)


def plot_field(
    grid:   np.ndarray,
    start:  Tuple[int, int],
    goal:   Tuple[int, int],
    path:   Optional[List[Tuple[int, int]]] = None,
    title:  str                             = "",
    hidden: Optional[List[Tuple[int, int]]] = None,
    explored: Optional[List[Tuple[int, int]]] = None,
    path_color: str                         = _PATH_COLOR,
    ax:     Optional[plt.Axes]              = None,
    show_legend: bool                       = True,
) -> plt.Axes:
    """
    Render the field grid with optional path, explored nodes, and markers.

    Parameters
    ----------
    grid        : Integer array of Cell codes.
    start, goal : (row, col) coordinates.
    path        : Ordered list of (row, col) cells from start to goal.
    title       : Axes title.
    hidden      : Hidden obstacle positions to outline (dashed rectangle).
    explored    : Cells visited during search (shaded lightly).
    path_color  : Colour for the path line.
    ax          : Existing axes to draw into; created if None.
    show_legend : Whether to add the legend (disable in tight subplots).

    Returns
    -------
    ax : The matplotlib Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    rows, cols = grid.shape

    # --- Base grid image ---
    ax.imshow(
        grid,
        cmap   = _CMAP,
        norm   = _NORM,
        origin = "upper",
        aspect = "equal",
    )

    # --- Grid lines ---
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="grey", linewidth=0.4, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # --- Axis labels (column = x, row = y) ---
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(np.arange(cols), fontsize=7)
    ax.set_yticklabels(np.arange(rows), fontsize=7)
    ax.set_xlabel("Column", fontsize=9)
    ax.set_ylabel("Row", fontsize=9)

    # --- Explored nodes (light blue wash) ---
    if explored:
        for (r, c) in explored:
            ax.add_patch(
                plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    color="cyan", alpha=0.25, zorder=1,
                )
            )

    # --- Path overlay ---
    if path and len(path) > 1:
        path_arr = np.array(path)
        ax.plot(
            path_arr[:, 1], path_arr[:, 0],
            color     = path_color,
            linewidth = 2.0,
            zorder    = 3,
        )
        ax.scatter(
            path_arr[1:-1, 1], path_arr[1:-1, 0],
            color  = path_color,
            s      = 18,
            zorder = 4,
        )

    # --- Hidden obstacle outlines (dashed, hot-pink) ---
    if hidden:
        for (r, c) in hidden:
            ax.add_patch(
                plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    fill      = False,
                    edgecolor = _HIDDEN_COLOR,
                    linewidth = 2.0,
                    linestyle = "--",
                    zorder    = 5,
                )
            )

    # --- Start marker ---
    sr, sc = start
    ax.plot(sc, sr, marker="o", markersize=12, color=_START_COLOR, zorder=6)
    ax.text(sc, sr, "S", ha="center", va="center", fontsize=7,
            fontweight="bold", color="white", zorder=7)

    # --- Goal marker ---
    gr, gc = goal
    ax.plot(gc, gr, marker="*", markersize=16, color=_GOAL_COLOR, zorder=6)
    ax.text(gc, gr, "G", ha="center", va="center", fontsize=7,
            fontweight="bold", color="black", zorder=7)

    # --- Legend ---
    legend_handles = list(_LEGEND_PATCHES)
    legend_handles.append(
        mpatches.Patch(facecolor="none", edgecolor=_HIDDEN_COLOR,
                       linestyle="--", label="Hidden obstacle (true map)")
    )
    if show_legend:
        ax.legend(
            handles   = legend_handles,
            loc       = "upper right",
            fontsize  = 7,
            framealpha= 0.8,
        )

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")

    return ax

