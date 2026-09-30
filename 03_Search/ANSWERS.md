# Lab 03: Written answers

## Task 0
State: (row, col) non-obstacle cell; Actions: Up/Down/Left/Right; Transition: apply the offset if inside the grid and not `#`; s0 = position of S = (1,1); Goal = position of G = (7,15); Cost = 1 per move.
(a) Only (row, col). (b) Leaving the grid or entering `#`. (c) Deterministic: one outcome per action. (d) A valid action sequence from s0 to the goal.

## Task 1 design
State = tuple; warehouse = list of lists; valid actions by bounds + `#` check; goal test on dequeue; frontier = heap of (f, tie-breaker, state) plus g and parent dictionaries; path by walking parents back. Report found / path / length / states expanded.

## Task 3 tests
1 original map: found, length 40, 63 expanded. 2 trivial: length 1. 3 no solution: both algorithms report failure after 9 expansions. 4 open room: A\* length 8 = BFS length 8 (shortest).

## Task 4
Concept map: state = tuples; action = `ACTIONS`; transition = `neighbours()`; goal test = `s == goal`; g = dict `g`; h = `manhattan()`; f = `ng + h(n, goal)`; frontier = `heapq` list; visited = `closed` set; path = `rebuild()`.
(a) Binary min-heap. (b) Pops the smallest f; ties by insertion order or by smaller h. (c) In `manhattan`, when a neighbour is pushed. (d) Yes, explicitly. (e) The closed set plus a push only when g improves.

## Task 5 (BFS vs A\*)
Both found length 40 and expanded 63 states, identical path. A\* can expand fewer because h steers away from far-looking cells, but this maze has one route and Manhattan distance points through walls, so there is nothing to prune. On an open room with low-h tie-breaking A\* expands 14 vs 59; with default tie-breaking it expands the same as BFS, so tie-breaking matters.

## Task 6 (heuristics)
Manhattan is admissible and consistent for 4-directional unit-cost moves. On the warehouse, h=0, Euclidean and 2xManhattan all give length 40 and 63 expansions. h=0 is blind search (optimal, slower); Euclidean is admissible but weaker; an inflated h (3x and 10x Manhattan on the extra map) expands fewer states (18 vs 27) but returns length 16 instead of the optimal 10. Too optimistic costs time; too aggressive risks optimality.

## Task 7 (evaluating the LLM-generated agent)
1. State/transition/goal/path code and the closed set were correct at once. 2. No algorithm bug; a design dependence on tie-breaking. 3. Found by comparing with BFS on an open room. 4. The tie-breaker counter and stale heap entries needed explaining. 5. A `prefer_low_h` flag and heuristic arguments were added afterwards. 6. The BFS cross-check and the no-solution test. 7. No, plausible paths are not proof. 8. A\* is only as good as its heuristic; tie-breaking matters; an inadmissible h can be suboptimal.

## Final reflection
1. Formulating first fixes states, actions, goal and cost, so code has something to be tested against.
2. A\* uses a heuristic, problem knowledge, to order the frontier by f = g + h.
3. Admissibility keeps optimality; informativeness prunes; here Manhattan gave no benefit in the maze.
4. The LLM contributed working code, tests and structure suggestions.
5. Untested code could return non-optimal paths, hang on unreachable goals or miscount expansions.
