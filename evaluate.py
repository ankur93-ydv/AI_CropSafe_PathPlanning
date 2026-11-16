"""
evaluate.py – Task 6: Performance Evaluation and Benchmarking
==============================================================
Runs offline algorithm comparisons (BFS, DFS, Greedy, A*) and online
replanning experiments across multiple obstacle scenarios.

Generates:
  - CSV files in results/
  - Markdown tables in results/results_tables.md
  - Matplotlib table renders in screenshots/
  - Bar charts in screenshots/06_bar_charts.png
  - 2x2 algorithm paths in screenshots/03_algorithm_paths.png
  - Explored nodes visualization in screenshots/03b_explored_nodes.png
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from field import Field, build_field, plot_field, SAFE, OBSTACLE
from problem import PathProblem
from search import bfs, dfs, greedy_best_first, a_star, manhattan, SearchResult
from replanning import run_replanning, ReplanResult


# ---------------------------------------------------------------------------
# Experiment A: Offline Comparison
# ---------------------------------------------------------------------------

def run_experiment_a(
    field_obj: Field,
    num_runs: int = 100,
) -> Tuple[pd.DataFrame, Dict[str, SearchResult]]:
    """
    Run BFS, DFS, Greedy, A* on the known_map (static offline map).
    Measures nodes expanded, path length, path cost, and average runtime over num_runs.
    """
    prob = PathProblem(field_obj.known_map, field_obj.start, field_obj.goal, use_penalty=True)

    algorithms = [
        ("BFS", lambda: bfs(prob)),
        ("DFS", lambda: dfs(prob)),
        ("Greedy Best-First", lambda: greedy_best_first(prob, manhattan)),
        ("A*", lambda: a_star(prob, manhattan)),
    ]

    results_dict: Dict[str, SearchResult] = {}
    rows = []

    for name, search_fn in algorithms:
        # Deterministic single run for path metrics
        res = search_fn()
        results_dict[name] = res

        # Benchmark runtime over num_runs
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = search_fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        mean_time = float(np.mean(times))
        std_time = float(np.std(times))

        rows.append({
            "Algorithm": name,
            "Path Length (steps)": res.path_length if res.reached_goal else "N/A",
            "Path Cost (penalised)": round(res.path_cost, 2) if res.reached_goal else "N/A",
            "Nodes Expanded": res.nodes_expanded,
            "Time Mean (ms)": round(mean_time, 4),
            "Time Std (ms)": round(std_time, 4),
            "Goal Reached": "Yes" if res.reached_goal else "No",
        })

    df = pd.DataFrame(rows)
    return df, results_dict


# ---------------------------------------------------------------------------
# Experiment B: Online Replanning Scenarios
# ---------------------------------------------------------------------------

def run_experiment_b(
    field_obj: Field,
) -> Tuple[pd.DataFrame, Dict[str, ReplanResult]]:
    """
    Run online replanning across 4 scenarios:
      1. Baseline: 0 hidden obstacles (no replanning needed)
      2. 1 hidden obstacle on path: (7, 6)
      3. 2 hidden obstacles: (7, 6) and (3, 3)
      4. Impossible goal: goal is completely surrounded by obstacles
    """
    start = field_obj.start
    goal = field_obj.goal
    base_map = field_obj.known_map.copy()

    scenarios = []

    # Scenario 1: Baseline
    scenarios.append((
        "1. Baseline (No hidden obstacles)",
        base_map.copy(),
        base_map.copy(),
        [],
    ))

    # Scenario 2: 1 hidden obstacle at (7, 6)
    map_1 = base_map.copy()
    map_1[7, 6] = OBSTACLE
    scenarios.append((
        "2. Single Obstacle at (7,6)",
        map_1,
        base_map.copy(),
        [(7, 6)],
    ))

    # Scenario 3: 2 hidden obstacles: (7, 6) and (3, 3)
    map_2 = base_map.copy()
    map_2[7, 6] = OBSTACLE
    map_2[3, 3] = OBSTACLE
    scenarios.append((
        "3. Multi Obstacles (7,6) & (3,3)",
        map_2,
        base_map.copy(),
        [(7, 6), (3, 3)],
    ))

    # Scenario 4: Impossible goal: surround (14, 14) at (13, 14) and (14, 13)
    map_imp = base_map.copy()
    map_imp[13, 14] = OBSTACLE
    map_imp[14, 13] = OBSTACLE
    scenarios.append((
        "4. Blocked Goal (Unreachable)",
        map_imp,
        base_map.copy(),
        [(13, 14), (14, 13)],
    ))

    replan_results: Dict[str, ReplanResult] = {}
    rows = []

    for name, true_m, known_m, hidden_list in scenarios:
        res = run_replanning(
            true_map=true_m,
            known_map_in=known_m,
            start=start,
            goal=goal,
            hidden_obs=hidden_list,
            scenario=name,
        )
        replan_results[name] = res

        rows.append({
            "Scenario": name,
            "Replans": res.replan_count,
            "Total Steps": res.total_steps,
            "Total Cost": round(res.total_cost, 2),
            "Total Nodes Expanded": res.total_nodes_expanded,
            "Total Time (ms)": round(res.total_time_ms, 3),
            "Goal Reached": "Yes" if res.reached_goal else "No",
        })

    df = pd.DataFrame(rows)
    return df, replan_results


# ---------------------------------------------------------------------------
# Visualisations and Exports
# ---------------------------------------------------------------------------

def save_table_image(
    df: pd.DataFrame,
    title: str,
    output_path: str,
) -> None:
    """Render a pandas DataFrame as a clean Matplotlib image table."""
    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 1.5), max(2.5, len(df) * 0.6 + 1.2)))
    ax.axis("off")
    ax.axis("tight")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2E4053")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if row % 2 == 0:
                cell.set_facecolor("#F2F4F4")
            else:
                cell.set_facecolor("#FFFFFF")

    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Table Render] Saved -> {output_path}")


def save_algorithm_paths_figure(
    field_obj: Field,
    results_dict: Dict[str, SearchResult],
    output_path: str,
) -> None:
    """Plot the 2x2 grid of algorithm paths."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    axes = axes.flatten()

    colors = {
        "BFS": "#1E90FF",
        "DFS": "#8A2BE2",
        "Greedy Best-First": "#FF8C00",
        "A*": "#008000",
    }

    for idx, (name, res) in enumerate(results_dict.items()):
        ax = axes[idx]
        title = (
            f"{name}\n"
            f"Steps: {res.path_length} | Cost: {res.path_cost:.1f} | "
            f"Nodes: {res.nodes_expanded}"
        )
        plot_field(
            grid=field_obj.known_map,
            start=field_obj.start,
            goal=field_obj.goal,
            path=res.path,
            title=title,
            path_color=colors.get(name, "#1E90FF"),
            ax=ax,
            show_legend=(idx == 0),
        )

    plt.suptitle("Comparison of Search Algorithm Paths (Offline Map)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Paths Plot] Saved -> {output_path}")


def save_explored_nodes_figure(
    field_obj: Field,
    results_dict: Dict[str, SearchResult],
    output_path: str,
) -> None:
    """Plot 2x2 grid showing explored search spaces."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(results_dict.items()):
        ax = axes[idx]
        title = f"{name} Exploration Space ({res.nodes_expanded} nodes)"
        plot_field(
            grid=field_obj.known_map,
            start=field_obj.start,
            goal=field_obj.goal,
            path=res.path,
            explored=res.explored,
            title=title,
            ax=ax,
            show_legend=(idx == 0),
        )

    plt.suptitle("Search Spaces Explored by Each Algorithm", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Explored Plot] Saved -> {output_path}")


def save_bar_charts(
    df_offline: pd.DataFrame,
    output_path: str,
) -> None:
    """Save comparison bar charts for nodes expanded and penalised cost."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    algos = df_offline["Algorithm"].tolist()
    nodes = df_offline["Nodes Expanded"].tolist()
    costs = df_offline["Path Cost (penalised)"].tolist()

    colors = ["#1E90FF", "#8A2BE2", "#FF8C00", "#2ECC71"]

    # Nodes expanded
    bars1 = ax1.bar(algos, nodes, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_title("Nodes Expanded (Lower is Better)", fontweight="bold")
    ax1.set_ylabel("Nodes Count")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{int(yval)}", ha="center", va="bottom", fontsize=9)

    # Path cost
    bars2 = ax2.bar(algos, costs, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_title("Penalised Path Cost (Lower is Safer/Better)", fontweight="bold")
    ax2.set_ylabel("Total Path Cost")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Bar Charts] Saved -> {output_path}")

