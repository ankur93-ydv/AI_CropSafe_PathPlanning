"""
replanning.py – Task 5: Online Obstacle Detection and Replanning
=================================================================
Simulates a robot navigating with partial knowledge of the field.
The robot maintains a `known_map`; as it moves it senses the cells
around it and updates the map if new obstacles are found.  Whenever
the planned path is blocked the robot invokes A* to replan.

Algorithm (exact loop implemented here)
----------------------------------------
1. current = start; known_map = initial map
2. plan = A*(known_map, current → goal)
3. while current != goal:
     a. sense surroundings (within sensor_range) using true_map
     b. if a newly detected obstacle changes known_map:
            log detection (step number, position, obstacle cell)
            update known_map
            if the remaining planned path is now blocked:
                replan_count += 1
                plan = A*(updated known_map, current → goal)
                if no path: stop, reached_goal=False
     c. move one step along the plan; record in trail
4. return result

The robot NEVER steps onto an obstacle: sensing always happens before
moving, and sensor_range ≥ 1 guarantees any adjacent obstacle is found
before the robot would walk into it.

Only replans when the CURRENT plan is blocked.  Obstacles detected that
do not lie on the remaining path are logged separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from field import Cell, SAFE, OBSTACLE
from problem import PathProblem
from search import SearchResult, a_star, manhattan

# ---------------------------------------------------------------------------
# Default sensor range
# ---------------------------------------------------------------------------
DEFAULT_SENSOR_RANGE: int = 2


# ---------------------------------------------------------------------------
# Result dataclass for the replanning simulation
# ---------------------------------------------------------------------------

@dataclass
class ReplanResult:
    """Full record of an online navigation run."""
    scenario:            str
    trail:               List[Tuple[int, int]]            # actual cells visited
    initial_plan:        List[Tuple[int, int]]
    plans_history:       List[List[Tuple[int, int]]]      # every plan computed
    detections:          List[Dict]                        # list of detection dicts
    replan_count:        int
    total_steps:         int
    total_cost:          float                             # penalised cost of trail
    total_nodes_expanded: int                              # sum across all A* calls
    total_time_ms:       float
    reached_goal:        bool
    final_known_map:     np.ndarray


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_replanning(
    true_map:     np.ndarray,
    known_map_in: np.ndarray,
    start:        Tuple[int, int],
    goal:         Tuple[int, int],
    hidden_obs:   List[Tuple[int, int]],
    sensor_range: int  = DEFAULT_SENSOR_RANGE,
    scenario:     str  = "default",
) -> ReplanResult:
    """
    Run the online navigation simulation.

    Parameters
    ----------
    true_map     : Ground-truth grid including hidden obstacles.
    known_map_in : Robot's initial belief map (without hidden obstacles).
    start, goal  : Start and goal positions.
    hidden_obs   : List of hidden obstacle positions (for logging only;
                   the simulation discovers them organically via sensing).
    sensor_range : Manhattan-distance sensing radius (default 2).
    scenario     : Descriptive name for this run.

    Returns
    -------
    ReplanResult with full navigation record.
    """
    # Work with copies so caller's maps are unaffected
    known_map    = known_map_in.copy()
    current      = start

    # ------------------------------------------------------------------
    # Initial plan
    # ------------------------------------------------------------------
    prob0  = PathProblem(known_map, current, goal, use_penalty=True)
    result0 = a_star(prob0, manhattan)
    if not result0.reached_goal:
        # Even the initial plan fails → no route at all
        return ReplanResult(
            scenario             = scenario,
            trail                = [current],
            initial_plan         = [],
            plans_history        = [],
            detections           = [],
            replan_count         = 0,
            total_steps          = 0,
            total_cost           = 0.0,
            total_nodes_expanded = result0.nodes_expanded,
            total_time_ms        = result0.time_ms,
            reached_goal         = False,
            final_known_map      = known_map,
        )

    plan:               List[Tuple[int, int]] = result0.path
    initial_plan        = list(plan)
    plans_history       = [list(plan)]
    trail               = [current]
    detections          = []
    replan_count        = 0
    total_nodes_expanded = result0.nodes_expanded
    total_time_ms        = result0.time_ms
    total_cost           = 0.0
    step_num             = 0
    reached_goal         = False

    # Penalty problem for cost tracking along the trail
    # We'll recompute when known_map changes
    cost_problem = PathProblem(known_map, current, goal, use_penalty=True)

    while current != goal:
        # ---------------------------------------------------------------
        # a. Sense surroundings
        # ---------------------------------------------------------------
        newly_detected = []
        cr, cc = current
        for r in range(max(0, cr - sensor_range),
                       min(true_map.shape[0], cr + sensor_range + 1)):
            for c in range(max(0, cc - sensor_range),
                           min(true_map.shape[1], cc + sensor_range + 1)):
                if abs(r - cr) + abs(c - cc) <= sensor_range:
                    if (true_map[r, c] == OBSTACLE
                            and known_map[r, c] != OBSTACLE):
                        newly_detected.append((r, c))

        # ---------------------------------------------------------------
        # b. Update known_map and decide whether to replan
        # ---------------------------------------------------------------
        if newly_detected:
            for obs in newly_detected:
                known_map[obs] = OBSTACLE
                or_c, or_obs = current, obs
                detections.append({
                    "step":          step_num,
                    "robot_pos":     or_c,
                    "obstacle_cell": or_obs,
                    "replanned":     False,          # updated below
                })

            # Check if the remaining plan is still passable
            # (plan[0] == current; remaining = plan[1:])
            remaining = plan[1:] if len(plan) > 1 else []
            blocked   = any(known_map[r, c] != SAFE for (r, c) in remaining)

            if blocked:
                replan_count += 1
                new_prob = PathProblem(known_map, current, goal, use_penalty=True)
                new_res  = a_star(new_prob, manhattan, start=current)
                total_nodes_expanded += new_res.nodes_expanded
                total_time_ms        += new_res.time_ms
                # mark detections that triggered this replan
                for d in detections[-len(newly_detected):]:
                    d["replanned"] = True

                if not new_res.reached_goal:
                    print(
                        f"[Step {step_num}] REPLAN #{replan_count} failed – "
                        "goal unreachable from current position!"
                    )
                    break

                plan = new_res.path
                plans_history.append(list(plan))
                cost_problem = new_prob

                print(
                    f"[Step {step_num}] Robot at {current} detected NEW obstacle(s) "
                    f"{newly_detected}. Path blocked → replanning (#{replan_count})"
                )
            else:
                print(
                    f"[Step {step_num}] Robot at {current} detected obstacle(s) "
                    f"{newly_detected} – path not blocked, no replan needed."
                )

        # ---------------------------------------------------------------
        # c. Move one step along the plan
        # ---------------------------------------------------------------
        if len(plan) < 2:
            # Should not happen if we reached goal; edge case
            break

        next_cell = plan[1]
        nr, nc    = next_cell

        # Safety check: never step onto a true obstacle
        if true_map[nr, nc] == OBSTACLE:
            print(
                f"[Step {step_num}] SAFETY ABORT: next cell {next_cell} "
                "is an obstacle in true_map! Check sensor_range."
            )
            break

        # Accumulate cost and move
        total_cost += cost_problem.step_cost(next_cell)
        current     = next_cell
        plan        = plan[1:]   # advance plan pointer
        trail.append(current)
        step_num   += 1

        if current == goal:
            reached_goal = True
            print(f"[Step {step_num}] Robot reached GOAL {goal}!")
            break

    return ReplanResult(
        scenario             = scenario,
        trail                = trail,
        initial_plan         = initial_plan,
        plans_history        = plans_history,
        detections           = detections,
        replan_count         = replan_count,
        total_steps          = len(trail) - 1,
        total_cost           = total_cost,
        total_nodes_expanded = total_nodes_expanded,
        total_time_ms        = total_time_ms,
        reached_goal         = reached_goal,
        final_known_map      = known_map,
    )


# ---------------------------------------------------------------------------
# Visualisation helpers (called from main.py)
# ---------------------------------------------------------------------------

def build_replanning_figure(
    field_obj,
    rr: ReplanResult,
) -> "matplotlib.figure.Figure":
    """
    Build a 4-panel figure:
      1. Before  – initial known map with initial plan.
      2. Detection – map at first obstacle discovery, old path shown dashed.
      3. After   – new plan after first replan.
      4. Final   – full actual trail with replan markers.

    Returns the Figure (caller saves it).
    """
    import matplotlib.pyplot as plt
    from field import plot_field, _CMAP_COLORS
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle("Task 5 – Online Obstacle Detection and Replanning", fontsize=13)

    start = field_obj.start
    goal  = field_obj.goal
    known_init = field_obj.known_map

    # ---- Panel 1: Before (initial known map + initial plan) ----
    ax1 = axes[0]
    plot_field(
        known_init, start, goal,
        path        = rr.initial_plan,
        title       = "1. Before (Initial Plan)",
        hidden      = field_obj.hidden_obstacles,
        ax          = ax1,
        show_legend = False,
    )

    # ---- Panel 2: At first detection ----
    ax2 = axes[1]
    first_det = rr.detections[0] if rr.detections else None
    if first_det and len(rr.plans_history) > 1:
        # Show intermediate known map (after first detection)
        # We reconstruct it: known_init + first obstacle only
        mid_known = known_init.copy()
        mid_known[first_det["obstacle_cell"]] = OBSTACLE
        plot_field(
            mid_known, start, goal,
            path        = rr.initial_plan,   # blocked path shown dashed-style
            title       = f"2. Detection at step {first_det['step']}",
            hidden      = [first_det["obstacle_cell"]],
            ax          = ax2,
            path_color  = "#FF4500",         # red = blocked path
            show_legend = False,
        )
        # Mark robot position at detection
        rp = first_det["robot_pos"]
        ax2.plot(rp[1], rp[0], "g^", markersize=12, zorder=8, label="Robot pos")
    else:
        plot_field(known_init, start, goal, title="2. No detection", ax=ax2,
                   show_legend=False)

    # ---- Panel 3: After replanning (second plan) ----
    ax3 = axes[2]
    second_plan = rr.plans_history[1] if len(rr.plans_history) > 1 else rr.initial_plan
    plot_field(
        rr.final_known_map, start, goal,
        path        = second_plan,
        title       = "3. After Replanning (New Path)",
        ax          = ax3,
        show_legend = False,
    )

    # ---- Panel 4: Full actual trail ----
    ax4 = axes[3]
    plot_field(
        rr.final_known_map, start, goal,
        path        = rr.trail,
        title       = f"4. Final Trail (replans={rr.replan_count})",
        hidden      = field_obj.hidden_obstacles,
        ax          = ax4,
        show_legend = True,
    )
    # Mark replan positions
    for d in rr.detections:
        if d["replanned"]:
            rp = d["robot_pos"]
            ax4.plot(rp[1], rp[0], "r*", markersize=12, zorder=8)

    plt.tight_layout()
    return fig


def build_animation(
    field_obj,
    rr: ReplanResult,
    output_path: str,
) -> None:
    """
    Save an animated GIF of the robot moving along its trail.
    Uses PillowWriter; wraps everything in try/except so it never
    crashes the main program.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from field import plot_field

        fig, ax = plt.subplots(figsize=(7, 7))
        trail   = rr.trail
        start   = field_obj.start
        goal    = field_obj.goal
        grid    = rr.final_known_map

        def _update(frame_idx: int):
            ax.clear()
            current_trail = trail[:frame_idx + 1]
            plot_field(
                grid, start, goal,
                path   = current_trail,
                title  = f"Step {frame_idx} – Robot at {trail[frame_idx]}",
                hidden = field_obj.hidden_obstacles,
                ax     = ax,
            )

        ani = animation.FuncAnimation(
            fig, _update,
            frames   = len(trail),
            interval = 400,
            repeat   = False,
        )
        writer = animation.PillowWriter(fps=2)
        ani.save(output_path, writer=writer)
        plt.close(fig)
        print(f"[Animation] Saved GIF → {output_path}")

    except Exception as exc:
        print(f"[Animation] Could not save GIF ({exc}). Skipping.")

