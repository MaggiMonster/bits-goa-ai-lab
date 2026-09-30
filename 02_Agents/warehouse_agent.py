"""Goal-based agent for the warehouse navigation problem.

The agent keeps a model of the warehouse (a 2-D grid), an explicit goal (cell G)
and a set of actions (Up, Down, Left, Right). To decide what to do it searches
ahead with Breadth-First Search (BFS) for a sequence of actions that reaches the
goal, then executes that plan.

Why BFS: every move costs 1, so BFS is complete (finds a path if one exists) and
optimal (returns a shortest path). Time/space are O(rows*cols) for this grid.
"""
from collections import deque

WAREHOUSE = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################",
]

# action name -> (row change, column change)
ACTIONS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


def parse(layout):
    """Return (grid, start, goal) from a list of strings."""
    grid = [list(row) for row in layout]
    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("map must contain both S and G")
    return grid, start, goal


def successors(grid, state):
    """Legal (action, next_state) pairs: stay on the grid and avoid '#'."""
    r, c = state
    for name, (dr, dc) in ACTIONS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != "#":
            yield name, (nr, nc)


def bfs(grid, start, goal):
    """Return the list of actions from start to goal, or None if unreachable."""
    frontier = deque([start])
    came_from = {start: None}  # state -> (previous state, action); also the visited set
    while frontier:
        state = frontier.popleft()
        if state == goal:  # goal test
            plan = []
            while came_from[state] is not None:
                state, action = came_from[state]
                plan.append(action)
            return plan[::-1]
        for action, nxt in successors(grid, state):
            if nxt not in came_from:
                came_from[nxt] = (state, action)
                frontier.append(nxt)
    return None


def follow(start, plan):
    """Execute a plan and return the visited cells (including start)."""
    cells = [start]
    r, c = start
    for a in plan:
        dr, dc = ACTIONS[a]
        r, c = r + dr, c + dc
        cells.append((r, c))
    return cells


def render(grid, cells):
    """Draw the path with '*' on top of the map."""
    out = [row[:] for row in grid]
    for r, c in cells[1:-1]:
        out[r][c] = "*"
    return "\n".join("".join(row) for row in out)


def run(layout):
    grid, start, goal = parse(layout)
    plan = bfs(grid, start, goal)
    if plan is None:
        print("No collision-free path exists from S to G.")
        return None
    cells = follow(start, plan)
    print(f"Path found: {len(plan)} moves")
    print("Actions:", " ".join(plan))
    print("Cells:", cells)
    print(render(grid, cells))
    return plan


if __name__ == "__main__":
    run(WAREHOUSE)
