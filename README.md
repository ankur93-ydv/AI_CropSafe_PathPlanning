# PROJECT SPECIFICATION: Potato-Field Robot Path Planning (AI Search)

> **Instructions to the AI coding agent:** Read this entire document, then build the complete project described below inside the current workspace. Create every file listed, run the program, fix any errors, generate the output screenshots into `screenshots/`, and fill the results tables in `README.md` with the **real numbers** produced by the program. Do not leave placeholders or TODOs. Ask no clarifying questions; where a choice is left open, use the defaults stated here.

---

## 1. Project Overview

This is a university **Artificial Intelligence group coding assignment**. We must build an intelligent path-planning agent that guides an autonomous robot from a start position to a destination in a simulated **potato field**, modelled as a **2D grid**.

The robot must:
- travel between crop rows without damaging plants,
- avoid obstacles (stones, tools, other objects) and crop-protection regions,
- choose routes with **low overall movement cost**, preferring cells farther from protected crops,
- **detect an initially unknown obstacle** during its journey, update its known map, and **replan using A\***.

We must implement and compare **BFS, DFS, Greedy Best-First Search, and A\***.

**Language:** Python 3.9+
**Libraries:** `numpy`, `matplotlib`, `pandas` (optional but preferred for tables). Use built-ins `heapq`, `collections`, `time`, `dataclasses`, `typing`. No other heavy dependencies.

---

## 2. Required Repository Structure

```
potato-robot-planner/
├── README.md              # full documentation (see Section 10)
├── requirements.txt       # numpy, matplotlib, pandas
├── main.py                # entry point: runs all tasks end to end
├── field.py               # Task 1: grid creation + visualisation
├── problem.py             # Task 2 + 4: problem formulation, costs, penalty map
├── search.py              # Task 3: BFS, DFS, Greedy, A*
├── replanning.py          # Task 5: online obstacle detection + replanning
├── evaluate.py            # Task 6: benchmarking + tables
├── screenshots/           # generated PNGs (see Section 9)
└── results/               # generated CSV/markdown tables
```

`python main.py` must run the whole project with no arguments, save all figures to `screenshots/`, save tables to `results/`, print tables to the console, and (optionally) show the plots. Use `matplotlib.use("Agg")` fallback when no display is available, and support a `--show` flag to display windows.

---

## 3. Task 1: Field Environment (`field.py`)

Build a 2D grid using a NumPy integer array. Recommended size: **15 rows × 15 columns** (must be at least 12×12).

### Cell codes (define as constants / an `IntEnum`)

| Code | Name | Meaning | Traversable? |
|---|---|---|---|
| 0 | `SAFE` | Safe travelling cell (soil aisle / headland) | Yes |
| 1 | `POTATO_ROW` | Planted potato row | **No** (robot must not trample crops) |
| 2 | `PROTECTED` | Crop-protection region (seedlings, irrigation, sensitive area) | **No** |
| 3 | `OBSTACLE` | Stone, tool or other object | **No** |

Start and goal are stored as coordinates `(row, col)`, not as cell codes, and must lie on `SAFE` cells.

### Layout requirements
- Potato rows are **vertical stripes** (e.g., 2 columns wide) separated by open aisles, with **horizontal cross-aisles** (headlands) at the top, bottom, and at least one middle row so the robot has several alternative routes.
- Include **at least two crop-protection regions** (small rectangular blocks, e.g., 2×2 or 1×3) placed so that some aisles run **right next to them**. This is what makes the proximity penalty change the route.
- Include **at least 3 known obstacles** (visible in the robot's initial map).
- Include **at least 2 hidden obstacles** (present in the `true_map` only, absent from the robot's `known_map`). At least one hidden obstacle **must lie on the original optimal A\* path**, so replanning is genuinely triggered. Place the second hidden obstacle so it triggers a **second** replan if possible.
- **Start** near the top-left, **goal** near the bottom-right (or similar), far enough apart that routes differ between algorithms.
- Provide a function that **validates the layout**: using BFS on the `true_map`, assert a path exists from start to goal even after all hidden obstacles are added. Raise a clear error otherwise.

### Data structures
Provide a `Field` class (or dataclass) with:
- `true_map: np.ndarray` (ground truth including hidden obstacles)
- `known_map: np.ndarray` (what the robot initially knows; copy of true map without hidden obstacles)
- `start: tuple[int,int]`, `goal: tuple[int,int]`
- `hidden_obstacles: list[tuple[int,int]]`
- helper methods: `in_bounds(r,c)`, `is_traversable(map, r,c)`, `copy_known_map()`.

### Visualisation function
`plot_field(grid, start, goal, path=None, title="", hidden=None, ax=None, ...)`
- Use `matplotlib` `imshow` with a custom `ListedColormap`:
  - SAFE = light beige/tan, POTATO_ROW = green, PROTECTED = red/orange, OBSTACLE = dark grey/black.
- Mark **Start** (blue marker with "S") and **Goal** (gold star with "G").
- Draw grid lines, axis ticks for row/col indices.
- Include a **legend** that explains every colour and marker.
- When `path` is provided, overlay it as a coloured line with small dots.
- Optionally show hidden obstacles (e.g., dashed outline or hatched) only when explicitly asked (for the replanning demo).
- Must be able to draw into a given `ax` so we can build subplots.

Save the base picture as `screenshots/01_field_environment.png`.

---

## 4. Task 2: Problem Formulation (`problem.py`)

Implement a `PathProblem` class that formally models the search problem. **Document the formulation in docstrings and in the README.**

- **State:** robot position `(row, col)`.
- **Initial state:** the start cell.
- **Actions:** `UP (-1,0)`, `DOWN (+1,0)`, `LEFT (0,-1)`, `RIGHT (0,+1)`. **4-connected movement only; diagonal moves are NOT allowed.** (State this explicitly in the README.)
- **Transition model:** `result(state, action)` returns the neighbouring cell if it is inside the grid and traversable (`SAFE`); otherwise the action is not applicable. `actions(state)` returns only the legal actions.
- **Goal test:** `is_goal(state)` is true when `state == goal`.
- **Path cost:** the sum of step costs, where
  `step_cost(next_state) = 1 + proximity_penalty(next_state)`.
  Base move cost is 1 per step.

The problem class takes a **map** as a parameter (so that it can run on `true_map`, or on the robot's evolving `known_map` for replanning) and recomputes/uses the penalty map accordingly.

Provide `successors(state)` returning a list of `(next_state, step_cost)`.

---

## 5. Task 3: Search Algorithms (`search.py`)

Implement **all four** algorithms from scratch (no search libraries). All must share one **identical interface** and return a common result object.

### Result dataclass
```python
@dataclass
class SearchResult:
    algorithm: str
    path: list[tuple[int,int]]   # empty list if not found
    path_length: int             # number of steps (len(path)-1), 0 or -1 if not found
    path_cost: float             # sum of step costs along the path (including penalties)
    nodes_expanded: int
    time_ms: float               # execution time in milliseconds (time.perf_counter)
    reached_goal: bool
    explored: list[tuple[int,int]]  # order of expanded nodes (for optional visualisation)
```

### Function signatures
```python
def bfs(problem) -> SearchResult
def dfs(problem) -> SearchResult
def greedy_best_first(problem, heuristic) -> SearchResult
def a_star(problem, heuristic) -> SearchResult
```
`problem` is a `PathProblem`; for replanning, `a_star` must also accept a different start state (e.g., `a_star(problem, heuristic, start=current_pos)`).

### Algorithm details
| Algorithm | Frontier | Priority | Notes |
|---|---|---|---|
| **BFS** | FIFO queue (`collections.deque`) | – | Explored set; complete; shortest in **number of steps**; ignores penalty costs. |
| **DFS** | LIFO stack | – | Explored set (graph search); complete on finite grid; **not optimal**; neighbour order must be fixed and documented (e.g., UP, RIGHT, DOWN, LEFT) so results are reproducible. |
| **Greedy Best-First** | `heapq` priority queue | `h(n)` | Not optimal, not guaranteed shortest; fast. Tie-break deterministically (counter). |
| **A\*** | `heapq` priority queue | `f(n) = g(n) + h(n)` | `g` uses the **penalised path cost**; keep best-known `g` per state; allow re-opening when a cheaper `g` is found (or use closed set correctly with a consistent heuristic). Optimal with respect to penalised cost. |

### Rules
- **Count `nodes_expanded`** consistently across all algorithms: increment when a node is **popped from the frontier and expanded** (i.e., its successors are generated). Document this definition.
- Keep a `parent` dict to reconstruct the path.
- Measure `time_ms` using `time.perf_counter()` around the search only.
- If no path exists, return `reached_goal=False`, empty path, without crashing.
- All algorithms respect obstacles and protected cells (they are never successors).

### Heuristic
`manhattan(state, goal) = |r - r_goal| + |c - c_goal|`

**Explain in the README and code comments:**
- It is the exact shortest distance on an empty 4-connected grid with unit step cost.
- It is **admissible**: it never overestimates the true remaining cost, because each step costs **at least 1** (penalties only add cost) and each move changes Manhattan distance by at most 1.
- It is **consistent** (monotone): `h(n) <= c(n,n') + h(n')` for every move, so A\* never needs to re-expand nodes and is optimal.
- Greedy uses only `h(n)`, so it is fast but ignores the cost so far and may take penalised or longer routes.

---

## 6. Task 4: Crop-Safe Planning (`problem.py`)

- **Hard constraint:** `POTATO_ROW`, `PROTECTED`, and `OBSTACLE` cells are never entered by any algorithm.
- **Soft constraint (proximity penalty):** build a `penalty_map` (NumPy float array, same shape as the grid) **once** per map using these defaults (make them configurable constants at the top of `problem.py`):
  - Safe cell at Manhattan distance **1** from any `PROTECTED` cell: **+5**
  - Safe cell at Manhattan distance **2** from any `PROTECTED` cell: **+2**
  - Otherwise: **+0**
  - (Optional, smaller: +1 for safe cells adjacent to a `POTATO_ROW` cell. If included, document it, and make sure it does not make the whole field uniformly penalised.)
  - Use the **maximum** applicable penalty if multiple apply, not the sum.
- Step cost into a cell = `1 + penalty_map[cell]`.
- **Demonstration required:** run A\* **twice** on the same map: (a) **without** penalties (all step costs = 1) and (b) **with** penalties. Show the two paths **side by side** in one figure (`screenshots/04_crop_safe_comparison.png`) along with a heat-map or shading of the penalty zones, and print both path lengths and path costs. The map must be designed so the two paths **actually differ** (the safe path is longer in steps but cheaper in penalised cost / farther from protected zones). Add a short printed explanation.
- Also report in the README the **minimum distance to protected cells** along each path, or the **number of penalised cells** visited, to prove the safer path is safer.

---

## 7. Task 5: Obstacle Detection and Replanning (`replanning.py`)

Simulate **online navigation** with partial knowledge.

### Setup
- `true_map` = ground truth including hidden obstacles.
- `known_map` = the robot's belief (initially **without** hidden obstacles).

### Sensing model (default)
- The robot senses all cells within **radius 1 (Manhattan) or the next cell(s) on its path, up to a configurable `sensor_range` (default 2)**. Make the sensing radius a parameter.
- Sensing reads from `true_map` only and writes discovered obstacles into `known_map`.

### Algorithm (implement exactly this loop)
```
1. current = start; known_map = initial map
2. plan = A*(known_map, current -> goal)
3. while current != goal:
     a. sense surroundings (within sensor_range) using true_map
     b. if a newly detected obstacle changes known_map:
           log detection (step number, position, obstacle cell)
           update known_map
           if the remaining planned path is now blocked:
               replan_count += 1
               plan = A*(updated known_map, current -> goal)
               if no path: stop, reached_goal=False
     c. move one step along the plan; record in `trail`
4. return result
```
- The robot must **never step onto** a cell that is an obstacle in the `true_map` (sensing happens before moving, so the sensor range ≥ 1 guarantees this).
- Only count a replan when the **current plan is actually blocked**; log separately obstacles that were detected but did not affect the route.
- Track: `trail` (actual cells travelled), `initial_plan`, `plans_history` (list of every plan computed), `detections` (list of dicts with step, robot position, obstacle cell), `replan_count`, `total_steps`, `total_cost` (penalised cost along the **actual trail**), `total_nodes_expanded` (sum across all A\* calls), `total_time_ms`, `reached_goal`.

### Visualisation (save as `screenshots/05_replanning_demo.png`, plus step-by-step frames if feasible)
A multi-panel figure showing at least:
1. **Before:** initial known map with the initially planned path (hidden obstacles not shown to the robot, but optionally outlined in a dashed style for the audience).
2. **At detection:** the robot's position when it first detects the hidden obstacle, the newly discovered obstacle highlighted, and the old blocked path shown (e.g., dashed red).
3. **After replanning:** the new path from the robot's current location.
4. **Final:** full actual trail from start to goal, with replan points marked.

Also save an **animated GIF** (`screenshots/05_replanning_demo.gif`) of the robot moving, if `matplotlib.animation.PillowWriter` is available. Wrap in try/except so failure does not crash the program.

Print a readable console log, e.g.:
```
[Step 6] Robot at (3, 1) detected NEW obstacle at (5, 1). Path blocked -> replanning (#1)
```

---

## 8. Task 6: Performance Evaluation (`evaluate.py`)

### Experiment A: Offline comparison (static known map = true map without hidden obstacles)
Run **BFS, DFS, Greedy, A\*** on the same field. Record:

| Algorithm | Path length (steps) | Path cost (penalised) | Nodes expanded | Time (ms) | Goal reached? |
|---|---|---|---|---|---|

- Time: run each algorithm **at least 100 times** and report the **mean** (and optionally std) execution time. Node counts and path are from a single deterministic run.
- Display the **path found by each algorithm** in a 2×2 subplot figure: `screenshots/03_algorithm_paths.png`, with each subplot titled with algorithm name, path length, cost, and nodes expanded. Optionally also overlay explored nodes in a light shade to show search effort (`screenshots/03b_explored_nodes.png`).

### Experiment B: Online obstacle experiment
Run the replanning simulation (Task 5) and report:

| Scenario | Replans | Total steps | Total cost | Total nodes expanded | Time (ms) | Goal reached? |
|---|---|---|---|---|---|---|

Include at least these scenarios: (1) no hidden obstacles (baseline, 0 replans), (2) one hidden obstacle on the path, (3) two or more hidden obstacles. Optionally (4) an impossible case in which the goal becomes unreachable, to show graceful failure (`reached_goal = False`).

### Experiment C (optional but valued)
Compare the same four algorithms on **2–3 additional maps** (different obstacle layouts), or compare A\* with and without penalties, and present a table.

### Output
- Print tables to the console (use `pandas.DataFrame.to_string()` or `tabulate`-style formatting, no extra dependency required).
- Save tables as `results/offline_comparison.csv`, `results/online_replanning.csv`, and `results/results_tables.md` (markdown format, ready to paste into the README).
- Save a picture of each table (`screenshots/06_comparison_table.png`, `screenshots/06_replanning_table.png`) using `matplotlib` table rendering.
- Optionally, add bar charts (nodes expanded, path cost) as `screenshots/06_bar_charts.png`.

---

## 9. Expected Program Output (Acceptance Checklist)

When `python main.py` runs, it must produce and save:

| File | Content |
|---|---|
| `screenshots/01_field_environment.png` | Field grid with rows, protected regions, obstacles, start, goal, legend |
| `screenshots/03_algorithm_paths.png` | 2×2 subplot: BFS, DFS, Greedy, A\* paths |
| `screenshots/04_crop_safe_comparison.png` | A\* without penalty vs with penalty, side by side |
| `screenshots/05_replanning_demo.png` | Before / detection / after / final trail |
| `screenshots/05_replanning_demo.gif` | (optional) animation |
| `screenshots/06_comparison_table.png` | Performance table image |
| `screenshots/06_replanning_table.png` | Online experiment table image |
| `results/*.csv`, `results/results_tables.md` | Tables |

Console output must include: the problem formulation summary, each algorithm's result, the replanning log, and both tables.

**Self-check before finishing (the agent must do this):**
- [ ] Program runs from a clean checkout with `pip install -r requirements.txt && python main.py` with no errors.
- [ ] A\* path cost <= every other algorithm's path cost (penalised) on the offline map. Assert this in code or tests.
- [ ] BFS path length (steps) is the minimum number of steps among all algorithms.
- [ ] All returned paths avoid non-traversable cells (write a `validate_path` helper and assert it).
- [ ] Replanning demo triggers **at least 1** replan (preferably 2), and the robot reaches the goal.
- [ ] The with-penalty and without-penalty A\* paths differ.
- [ ] No placeholder text in the README; the tables contain real numbers.
- [ ] Results are deterministic (fix any random seeds; no randomness in the core search).

Add a small `tests/test_search.py` (using `pytest` or plain asserts) covering: path validity, BFS optimality in steps, A\* optimality in cost, goal-unreachable handling, and replanning triggering. Mention how to run it in the README.

---

## 10. README.md Requirements

The `README.md` must be complete and polished, with these sections:

1. **Title and project overview**: what the project does and why.
2. **Team members** table (placeholder rows: `Name | Roll No. | Contribution`; leave clearly marked for the students to fill in, this is the only allowed placeholder).
3. **Features / tasks covered** (Tasks 1–6 mapped to files).
4. **Project structure** (tree).
5. **Requirements and installation**: Python version, libraries, `pip install -r requirements.txt`.
6. **How to run**: `python main.py`, optional `--show`, how to run tests.
7. **Field environment**: cell legend, grid size, explanation of rows, protected regions, obstacles, hidden obstacles; embed `screenshots/01_field_environment.png`.
8. **Problem formulation**: initial state, actions (4-connected, no diagonals), transition model, goal test, path cost function (state the formula and the penalty values).
9. **Algorithms**: short explanation of BFS, DFS, Greedy, A\* (data structure, completeness, optimality, complexity) and how each was implemented.
10. **Heuristic**: Manhattan distance, definition, admissibility and consistency proofs (brief), and why it suits a 4-connected grid.
11. **Crop-safe planning**: penalty design, rationale, side-by-side figure and discussion.
12. **Obstacle detection and replanning**: sensor model, replanning loop, figures, example log.
13. **Performance evaluation**: both result tables (real numbers) and embedded screenshots.
14. **Observations and discussion** (see below).
15. **Limitations and future work** (e.g., diagonal moves, dynamic obstacles, D\* Lite, weighted A\*, larger fields, real sensors).
16. **Screenshots** section with all output images embedded via relative paths.
17. **Conclusion**.

### Observations to cover in the discussion (use actual measured results to support each)
- BFS finds the fewest-step path but ignores crop-safety penalties, so it may be costlier in penalised terms.
- DFS is complete here but returns long, winding, non-optimal paths; node expansion can be low or high depending on neighbour ordering.
- Greedy is usually fast (few expansions) but not optimal and can walk near protected zones.
- A\* returns the lowest-cost (safest, shortest) route with fewer expansions than BFS thanks to the heuristic.
- Effect of the proximity penalty on path shape.
- Replanning costs additional node expansions but allows recovery from unknown obstacles; replan count vs. number of hidden obstacles.
- Trade-offs between optimality, speed, and memory.

---

## 11. Code Quality Requirements

- Clean, modular, **PEP 8** style, with **type hints** and **docstrings** on every public function/class.
- Meaningful names; constants for cell codes, penalties, move order, sensor range.
- Comments explaining the logic of each algorithm and the heuristic (students must be able to explain the code in a viva).
- No global mutable state; pass maps and problems as arguments.
- Deterministic behaviour.
- Handle edge cases: start == goal, goal unreachable, start or goal on a non-traversable cell (raise a clear error).
- `requirements.txt` with pinned minimum versions, e.g.:
  ```
  numpy>=1.23
  matplotlib>=3.6
  pandas>=1.5
  pytest>=7.0
  ```
- Add a `.gitignore` (Python defaults; **do not** ignore `screenshots/` or `results/`, they must be committed).

---

## 12. Suggested Example Map (use as a starting point; validate and adjust)

15×15, `.` = SAFE, `#` = POTATO_ROW, `P` = PROTECTED, `X` = known OBSTACLE, `h` = hidden obstacle (true map only), `S` = start, `G` = goal.

```
     0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
 0   S . . . . . . . . . .  .  .  .  .
 1   . # # . # # . # # . #  #  .  .  .
 2   . # # . # # . # # . #  #  .  P  P
 3   . # # . # # . # # . #  #  .  P  P
 4   . # # . # # . # # . #  #  .  .  .
 5   . # # . # # X # # . #  #  .  .  .
 6   . # # . # # . # # . #  #  .  .  .
 7   . . . . . . . . . . .  .  .  .  .
 8   . # # . # # . # # . #  #  .  .  .
 9   . # # . # # . # # X #  #  .  P  P
10   . # # . # # . # # . #  #  .  P  P
11   . # # . # # . # # . #  #  .  .  .
12   . # # . # # . # # . #  #  .  .  .
13   . # # . # # . # # . #  #  .  .  .
14   . . . . . . . . . . .  .  .  .  G
```
This is only a template: the agent must **adjust** it (add protected blocks adjacent to aisles, place hidden obstacles such as at `(10, 3)` or `(7, 6)` or another cell on the original A\* path, add a third known obstacle, ensure alternate routes exist) so that every acceptance criterion in Section 9 is satisfied. After finalising, **verify programmatically** that: a path exists, the penalised and un-penalised A\* paths differ, at least one hidden obstacle blocks the original A\* path, and the goal remains reachable after all hidden obstacles are revealed.

---

## 13. Deliverables Summary (for GitHub submission)

- Complete Python source code (all files in Section 2).
- `README.md` as specified in Section 10.
- `screenshots/` with all generated output images.
- Results tables (in README, CSV, and as images) with a **brief summary of observations**.
- Repository must be accessible to the instructor (students will handle repo creation and permissions; the agent should also provide the git commands to initialise, commit and push in a final message).

---

## 14. Final Instructions to the Agent

1. Create all files and the folder structure.
2. Install requirements and run `python main.py`; run the tests.
3. Fix every error until everything passes and all images/tables are produced.
4. Fill `README.md` with the real results and embed the screenshots.
5. In your final message, give a short summary: what was built, how to run it, the key results (the comparison table), and the commands to push to GitHub.
