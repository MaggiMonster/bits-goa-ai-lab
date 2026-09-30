# Lab 02: Written answers

## Task 1
1. **Environment:** 7x21 grid; `#` impassable, `.` free; fully observable, static, deterministic, discrete, single agent.
2. **Goal:** reach G from S by a collision-free path.
3. **Actions:** Up, Down, Left, Right (legal only inside the grid and not into `#`).
4. **Information kept:** the map, current position, goal position, and search bookkeeping (frontier, visited set, parent links).
5. **Goal-based, not reflex:** it has an explicit goal and searches over future action sequences; a reflex rule ("go right if free") gets stuck in dead ends.
- *Think about it, twice as large:* BFS stays correct and optimal but memory and time grow with the number of cells; A\* with Manhattan distance scales better. New difficulties: moving obstacles, partial observability, non-uniform costs, replanning.

## Task 2 design
Environment = grid; state = (row, col); goal = cell G; actions = four moves; decision maker = BFS planner. Block diagram is in the report.

## Task 3 questions
1. *Working first attempt?* Yes; it ran and passed all tests with no edits.
2. *Improving the prompt:* give the exact map, ask for validity/optimality/no-path tests, fix the output format.
3. *Algorithm:* breadth-first search.
4. *Why:* unit move costs and a tiny state space make BFS complete, optimal and simple.

## Strengths and limits of LLM-assisted development
Strength: fast, documented code from a precise spec. Limit: plausible output is not proof; independent tests (obstacle check, optimality check, unreachable case) establish correctness, and the algorithm choice must be checked for fit and scale.
