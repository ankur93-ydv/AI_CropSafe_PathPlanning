# Benchmarking and Evaluation Results

## 1. Offline Search Algorithms Comparison (Static Map)

| Algorithm         | Path Length (steps) | Path Cost (penalised) | Nodes Expanded | Time Mean (ms) | Time Std (ms) | Goal Reached |
| ----------------- | ------------------- | --------------------- | -------------- | -------------- | ------------- | ------------ |
| BFS               | 28                  | 68.0                  | 118            | 2.8068         | 0.7383        | Yes          |
| DFS               | 42                  | 124.0                 | 43             | 0.8831         | 0.2307        | Yes          |
| Greedy Best-First | 32                  | 96.0                  | 38             | 0.7064         | 0.0979        | Yes          |
| A*                | 28                  | 28.0                  | 80             | 2.1172         | 0.422         | Yes          |

## 2. Online Obstacle Replanning Scenarios

| Scenario                          | Replans | Total Steps | Total Cost | Total Nodes Expanded | Total Time (ms) | Goal Reached |
| --------------------------------- | ------- | ----------- | ---------- | -------------------- | --------------- | ------------ |
| 1. Baseline (No hidden obstacles) | 0       | 28          | 28.0       | 80                   | 1.884           | Yes          |
| 2. Single Obstacle at (7,6)       | 1       | 36          | 36.0       | 114                  | 3.373           | Yes          |
| 3. Multi Obstacles (7,6) & (3,3)  | 2       | 32          | 46.0       | 205                  | 4.494           | Yes          |
| 4. Blocked Goal (Unreachable)     | 2       | 27          | 27.0       | 202                  | 8.209           | No           |

## 3. Crop-Safe Planning Impact Summary

| Condition                 | Steps | Penalised Cost | Min Dist to Protected | Penalised Cells Visited |
| ------------------------- | ----- | -------------- | --------------------- | ----------------------- |
| A* Without Penalty        | 28    | 68.0           | 1                     | 11                      |
| A* With Proximity Penalty | 28    | 28.0           | 4                     | 0                       |
