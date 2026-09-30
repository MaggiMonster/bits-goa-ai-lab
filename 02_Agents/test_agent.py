from warehouse_agent import *

grid, s, g = parse(WAREHOUSE)
plan = bfs(grid, s, g)
cells = follow(s, plan)
assert cells[-1] == g
assert all(grid[r][c] != "#" for r, c in cells), "path crosses an obstacle"
assert all(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1 for a, b in zip(cells, cells[1:]))
print("path valid, length", len(plan))

# independent check of optimality: relax distances with Bellman-Ford style sweep
INF = 10**9
d = {(r, c): INF for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != "#"}
d[s] = 0
changed = True
while changed:
    changed = False
    for (r, c), v in list(d.items()):
        for _, n in successors(grid, (r, c)):
            if v + 1 < d[n]:
                d[n] = v + 1
                changed = True
assert d[g] == len(plan)
print("BFS length equals independent shortest distance:", d[g])

# unreachable goal -> message, None
blocked = ["#####", "#S#G#", "#####"]
assert run(blocked) is None
print("no-path case handled")
