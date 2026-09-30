"""A* and BFS for the warehouse navigation problem (grid, unit move cost).

State    : (row, col) tuple.
Actions  : Up, Down, Left, Right.
Transition: move one cell if it is inside the grid and not '#'.
Goal test: state == goal cell.
A*       : frontier = min-heap of (f, tie_breaker, state); f = g + h.
           'closed' set stops a state being expanded twice.
BFS      : FIFO queue; same interface and same bookkeeping.
"""
import heapq
import itertools
import math
from collections import deque

WAREHOUSE = [
    "#################",
    "#S....#.........#",
    "#.###.#.#######.#",
    "#...#.#.......#.#",
    "###.#.#######.#.#",
    "#...#.........#.#",
    "#.###########.#.#",
    "#.............#G#",
    "#################",
]

ACTIONS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


def parse(layout):
    grid = [list(r) for r in layout]
    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    return grid, start, goal


def neighbours(grid, s):
    """Valid (action, next_state) pairs."""
    r, c = s
    for a, (dr, dc) in ACTIONS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != "#":
            yield a, (nr, nc)


# heuristics: h(state, goal)
def manhattan(s, g): return abs(s[0] - g[0]) + abs(s[1] - g[1])
def zero(s, g): return 0
def euclidean(s, g): return math.hypot(s[0] - g[0], s[1] - g[1])
def double_manhattan(s, g): return 2 * manhattan(s, g)


def rebuild(parent, s):
    path = []
    while s is not None:
        path.append(s)
        s = parent[s]
    return path[::-1]


def astar(grid, start, goal, h=manhattan, prefer_low_h=False):
    """Return dict(found, path, length, expanded)."""
    tie = itertools.count()
    # heap entries: (f, secondary, tie, state); secondary = h when prefer_low_h (break f-ties toward goal)
    frontier = [(h(start, goal), 0, next(tie), start)]
    g = {start: 0}
    parent = {start: None}
    closed = set()
    expanded = 0
    while frontier:
        f, _, _, s = heapq.heappop(frontier)          # lowest f first
        if s in closed:
            continue                                   # stale queue entry
        if s == goal:                                  # goal test on expansion
            path = rebuild(parent, s)
            return dict(found=True, path=path, length=len(path) - 1, expanded=expanded)
        closed.add(s)
        expanded += 1
        for _, n in neighbours(grid, s):
            ng = g[s] + 1                              # g(n), unit cost
            if n not in g or ng < g[n]:
                g[n] = ng
                parent[n] = s
                hn = h(n, goal)
                heapq.heappush(frontier, (ng + hn, hn if prefer_low_h else 0, next(tie), n))   # f = g + h
    return dict(found=False, path=None, length=None, expanded=expanded)


def bfs(grid, start, goal):
    frontier = deque([start])
    parent = {start: None}
    expanded = 0
    while frontier:
        s = frontier.popleft()
        if s == goal:
            path = rebuild(parent, s)
            return dict(found=True, path=path, length=len(path) - 1, expanded=expanded)
        expanded += 1
        for _, n in neighbours(grid, s):
            if n not in parent:
                parent[n] = s
                frontier.append(n)
    return dict(found=False, path=None, length=None, expanded=expanded)


def show(res):
    if not res["found"]:
        print("No solution found. States expanded:", res["expanded"])
        return
    print("Solution found. Length:", res["length"], " States expanded:", res["expanded"])
    print("Path:", res["path"])


if __name__ == "__main__":
    grid, s, g = parse(WAREHOUSE)
    print("--- A* (Manhattan)"); show(astar(grid, s, g))
    print("--- BFS"); show(bfs(grid, s, g))
