"""
main.py – Complete Potato-Field Robot Path Planning Pipeline
============================================================
Entry point running all tasks end-to-end:
  Task 1: Environment generation & validation
  Task 2: Formal problem formulation
  Task 3: Search algorithms execution (BFS, DFS, Greedy, A*)
  Task 4: Crop-safe path planning comparison (with vs without penalty)
  Task 5: Online obstacle sensing and dynamic A* replanning
  Task 6: Benchmarking, evaluation tables, and visual exports

Usage:
  python main.py          # Runs headless and saves all screenshots and tables
  python main.py --show   # Also displays interactive visualization windows
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from field import (
    Field,
    build_field,
    plot_field,
    validate_layout,
    SAFE,
    POTATO_ROW,
    PROTECTED,
    OBSTACLE,
)
from problem import PathProblem, build_penalty_map, PENALTY_D1, PENALTY_D2
from search import (
    bfs,
    dfs,
    greedy_best_first,
    a_star,
    manhattan,
    validate_path,
    SearchResult,
)
from replanning import (
    run_replanning,
    build_replanning_figure,
    build_animation,
    ReplanResult,
)
from evaluate import (
    run_experiment_a,
    run_experiment_b,
    save_table_image,
    save_algorithm_paths_figure,
    save_explored_nodes_figure,
    save_bar_charts,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Potato-Field Robot Path Planner")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display matplotlib visualization windows interactively.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Configure backend
    if not args.show:
        matplotlib.use("Agg")

    # Ensure output directories exist
    base_dir = Path(__file__).resolve().parent
    screenshots_dir = base_dir / "screenshots"
    results_dir = base_dir / "results"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("  POTATO-FIELD ROBOT PATH PLANNING: AI SEARCH & REPLANNING PIPELINE")
    print("=" * 75)

    # -----------------------------------------------------------------------
    # Task 1: Field Environment
    # -----------------------------------------------------------------------
    print("\n[TASK 1] Initializing and Validating Field Environment...")
    field_obj = build_field()
    validate_layout(field_obj)
    print(f"  Field size       : {field_obj.true_map.shape[0]}x{field_obj.true_map.shape[1]}")
    print(f"  Start coordinate : {field_obj.start} (SAFE)")
    print(f"  Goal coordinate  : {field_obj.goal} (SAFE)")
    print(f"  Hidden obstacles : {field_obj.hidden_obstacles}")

    # Save Task 1 base figure
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    plot_field(
        grid=field_obj.known_map,
        start=field_obj.start,
        goal=field_obj.goal,
        title="Potato-Field Simulation Environment (15x15)",
        hidden=field_obj.hidden_obstacles,
        ax=ax1,
        show_legend=True,
    )
    fig1_path = str(screenshots_dir / "01_field_environment.png")
    fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved environment plot -> {fig1_path}")

    # -----------------------------------------------------------------------
    # Task 2: Problem Formulation
    # -----------------------------------------------------------------------
    print("\n[TASK 2] Problem Formulation Summary:")
    prob_penalised = PathProblem(field_obj.known_map, field_obj.start, field_obj.goal, use_penalty=True)
    print(prob_penalised.summary())

    # -----------------------------------------------------------------------
    # Task 3 & 6: Search Algorithms & Offline Benchmark (Experiment A)
    # -----------------------------------------------------------------------
    print("\n[TASK 3 & 6] Running Offline Search Algorithm Benchmark (100 runs each)...")
    df_offline, offline_results = run_experiment_a(field_obj, num_runs=100)

    print("\n" + df_offline.to_string(index=False))

    # Save offline comparison results
    csv_offline_path = str(results_dir / "offline_comparison.csv")
    df_offline.to_csv(csv_offline_path, index=False)
    print(f"\n  Saved offline CSV -> {csv_offline_path}")

    img_table_a_path = str(screenshots_dir / "06_comparison_table.png")
    save_table_image(df_offline, "Offline Search Algorithms Comparison (100 Runs Mean)", img_table_a_path)

    img_paths_path = str(screenshots_dir / "03_algorithm_paths.png")
    save_algorithm_paths_figure(field_obj, offline_results, img_paths_path)

    img_explored_path = str(screenshots_dir / "03b_explored_nodes.png")
    save_explored_nodes_figure(field_obj, offline_results, img_explored_path)

    img_bar_path = str(screenshots_dir / "06_bar_charts.png")
    save_bar_charts(df_offline, img_bar_path)

    # -----------------------------------------------------------------------
    # Task 4: Crop-Safe Planning Comparison (With vs Without Penalties)
    # -----------------------------------------------------------------------
    print("\n[TASK 4] Crop-Safe Planning Demonstration (A* With vs Without Proximity Penalty)...")
    prob_unpenalised = PathProblem(field_obj.known_map, field_obj.start, field_obj.goal, use_penalty=False)

    res_no_pen = a_star(prob_unpenalised, manhattan)
    res_with_pen = a_star(prob_penalised, manhattan)

    # Evaluate both paths under true penalised cost function
    cost_no_pen_eval = prob_penalised.path_cost(res_no_pen.path)
    cost_with_pen_eval = res_with_pen.path_cost

    # Calculate minimum distance to protected cells along each path
    protected_coords = list(zip(*np.where(field_obj.known_map == PROTECTED)))

    def min_dist_to_protected(path):
        min_d = float("inf")
        for (r, c) in path:
            for (pr, pc) in protected_coords:
                d = abs(r - pr) + abs(c - pc)
                if d < min_d:
                    min_d = d
        return min_d

    def count_penalised_cells(path):
        pen_map = prob_penalised.penalty_map
        return sum(1 for (r, c) in path if pen_map[r, c] > 0)

    min_d_no = min_dist_to_protected(res_no_pen.path)
    min_d_yes = min_dist_to_protected(res_with_pen.path)
    pen_count_no = count_penalised_cells(res_no_pen.path)
    pen_count_yes = count_penalised_cells(res_with_pen.path)

    print(f"  Without Penalty A*: Steps={res_no_pen.path_length}, Penalised Cost={cost_no_pen_eval:.1f}, "
          f"Min Dist to Protected={min_d_no}, Penalised Cells Visited={pen_count_no}")
    print(f"  With Penalty A*   : Steps={res_with_pen.path_length}, Penalised Cost={cost_with_pen_eval:.1f}, "
          f"Min Dist to Protected={min_d_yes}, Penalised Cells Visited={pen_count_yes}")

    # Generate side-by-side figure
    fig4, (ax4_1, ax4_2) = plt.subplots(1, 2, figsize=(15, 7))

    # Subplot 1: Without penalty
    plot_field(
        grid=field_obj.known_map,
        start=field_obj.start,
        goal=field_obj.goal,
        path=res_no_pen.path,
        title=f"A* Without Penalty\nSteps: {res_no_pen.path_length} | Penalised Cost: {cost_no_pen_eval:.1f}\nMin Dist to Protected: {min_d_no}",
        path_color="#E74C3C",
        ax=ax4_1,
        show_legend=True,
    )

    # Subplot 2: With penalty + penalty shading
    plot_field(
        grid=field_obj.known_map,
        start=field_obj.start,
        goal=field_obj.goal,
        path=res_with_pen.path,
        title=f"A* With Crop-Safety Penalty\nSteps: {res_with_pen.path_length} | Penalised Cost: {cost_with_pen_eval:.1f}\nMin Dist to Protected: {min_d_yes} (Safe Routing)",
        path_color="#2ECC71",
        ax=ax4_2,
        show_legend=False,
    )
    # Highlight penalty heat overlay on right plot
    pen_overlay = prob_penalised.penalty_map
    for r in range(pen_overlay.shape[0]):
        for c in range(pen_overlay.shape[1]):
            val = pen_overlay[r, c]
            if val == PENALTY_D1:
                ax4_2.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#E67E22", alpha=0.35, zorder=2))
            elif val == PENALTY_D2:
                ax4_2.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#F39C12", alpha=0.20, zorder=2))

    fig4_path = str(screenshots_dir / "04_crop_safe_comparison.png")
    fig4.savefig(fig4_path, dpi=200, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved crop-safe comparison plot -> {fig4_path}")

    # -----------------------------------------------------------------------
    # Task 5: Online Obstacle Sensing & Replanning Demonstration
    # -----------------------------------------------------------------------
    print("\n[TASK 5] Running Online Navigation and Replanning Simulation...")
    replan_demo_res = run_replanning(
        true_map=field_obj.true_map,
        known_map_in=field_obj.known_map,
        start=field_obj.start,
        goal=field_obj.goal,
        hidden_obs=field_obj.hidden_obstacles,
        sensor_range=2,
        scenario="Online Multi-Obstacle Replanning",
    )

    print(f"  Replanning Completed: {replan_demo_res.replan_count} replans triggered.")
    print(f"  Total Steps: {replan_demo_res.total_steps}, Total Cost: {replan_demo_res.total_cost:.1f}, "
          f"Nodes Expanded: {replan_demo_res.total_nodes_expanded}, Goal Reached: {replan_demo_res.reached_goal}")

    # Save Task 5 4-panel demo figure
    fig5 = build_replanning_figure(field_obj, replan_demo_res)
    fig5_path = str(screenshots_dir / "05_replanning_demo.png")
    fig5.savefig(fig5_path, dpi=200, bbox_inches="tight")
    plt.close(fig5)
    print(f"  Saved replanning multi-panel plot -> {fig5_path}")

    # Save Task 5 animation GIF
    gif5_path = str(screenshots_dir / "05_replanning_demo.gif")
    build_animation(field_obj, replan_demo_res, gif5_path)

    # -----------------------------------------------------------------------
    # Task 6: Online Replanning Experiment (Experiment B)
    # -----------------------------------------------------------------------
    print("\n[TASK 6] Running Online Replanning Multi-Scenario Evaluation...")
    df_online, online_results = run_experiment_b(field_obj)
    print("\n" + df_online.to_string(index=False))

    csv_online_path = str(results_dir / "online_replanning.csv")
    df_online.to_csv(csv_online_path, index=False)
    print(f"\n  Saved online replanning CSV -> {csv_online_path}")

    img_table_b_path = str(screenshots_dir / "06_replanning_table.png")
    save_table_image(df_online, "Online Replanning Benchmark Across Scenarios", img_table_b_path)

    # Save markdown summary tables to results/results_tables.md
    def df_to_markdown_str(df_in: pd.DataFrame) -> str:
        headers = list(df_in.columns)
        col_widths = [max(len(str(h)), max((len(str(v)) for v in df_in[h]), default=0)) for h in headers]
        header_row = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |"
        sep_row = "| " + " | ".join("-" * w for w in col_widths) + " |"
        data_rows = [
            "| " + " | ".join(f"{str(val):<{w}}" for val, w in zip(row, col_widths)) + " |"
            for row in df_in.values
        ]
        return "\n".join([header_row, sep_row] + data_rows)

    md_tables_path = str(results_dir / "results_tables.md")
    with open(md_tables_path, "w", encoding="utf-8") as f:
        f.write("# Benchmarking and Evaluation Results\n\n")
        f.write("## 1. Offline Search Algorithms Comparison (Static Map)\n\n")
        f.write(df_to_markdown_str(df_offline))
        f.write("\n\n")
        f.write("## 2. Online Obstacle Replanning Scenarios\n\n")
        f.write(df_to_markdown_str(df_online))
        f.write("\n\n")
        f.write("## 3. Crop-Safe Planning Impact Summary\n\n")
        crop_safe_df = pd.DataFrame([
            {
                "Condition": "A* Without Penalty",
                "Steps": res_no_pen.path_length,
                "Penalised Cost": round(cost_no_pen_eval, 2),
                "Min Dist to Protected": min_d_no,
                "Penalised Cells Visited": pen_count_no,
            },
            {
                "Condition": "A* With Proximity Penalty",
                "Steps": res_with_pen.path_length,
                "Penalised Cost": round(cost_with_pen_eval, 2),
                "Min Dist to Protected": min_d_yes,
                "Penalised Cells Visited": pen_count_yes,
            },
        ])
        f.write(df_to_markdown_str(crop_safe_df))
        f.write("\n")
    print(f"  Saved markdown tables -> {md_tables_path}")

    # -----------------------------------------------------------------------
    # Acceptance Criteria Self-Checks
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  VERIFYING ACCEPTANCE CRITERIA...")
    print("=" * 75)

    # 1. A* optimality in cost
    astar_cost = offline_results["A*"].path_cost
    for alg_name, res in offline_results.items():
        assert astar_cost <= res.path_cost + 1e-6, (
            f"Optimality violation: A* cost ({astar_cost}) > {alg_name} cost ({res.path_cost})"
        )
    print("  [PASS] A* achieves the lowest penalised path cost across all algorithms.")

    # 2. BFS optimality in step count
    bfs_steps = offline_results["BFS"].path_length
    for alg_name, res in offline_results.items():
        assert bfs_steps <= res.path_length, (
            f"Step optimality violation: BFS steps ({bfs_steps}) > {alg_name} steps ({res.path_length})"
        )
    print("  [PASS] BFS achieves the minimal step count across all algorithms.")

    # 3. Path validity
    for alg_name, res in offline_results.items():
        assert validate_path(res.path, prob_penalised), f"Path invalid for {alg_name}"
    print("  [PASS] All generated paths are contiguous and strictly traversable.")

    # 4. Replanning triggers >= 1 replan and succeeds
    assert replan_demo_res.replan_count >= 1, "Replanning was not triggered!"
    assert replan_demo_res.reached_goal is True, "Replanning failed to reach the goal!"
    print(f"  [PASS] Dynamic replanning triggered {replan_demo_res.replan_count} replans and safely reached goal.")

    # 5. Crop-safe difference
    assert res_no_pen.path != res_with_pen.path, "Penalised and unpenalised paths are identical!"
    assert cost_with_pen_eval < cost_no_pen_eval, "Safe path penalised cost is not strictly cheaper!"
    print("  [PASS] Crop-safe penalty shifts the path away from protected zones.")

    print("\nALL ACCEPTANCE CRITERIA PASSED SUCCESSFULLY!")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())

