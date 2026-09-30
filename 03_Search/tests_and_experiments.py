from search_agent import *

def run(layout, fn, **kw):
    g, s, t = parse(layout)
    return fn(g, s, t, **kw)

def valid(layout, res):
    g, s, t = parse(layout)
    p = res["path"]
    assert p[0] == s and p[-1] == t
    assert all(g[r][c] != "#" for r, c in p)
    assert all(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1 for a, b in zip(p, p[1:]))
    assert res["length"] == len(p) - 1

print("=== Test 1: original warehouse ===")
r = run(WAREHOUSE, astar); valid(WAREHOUSE, r); show(r)

print("\n=== Test 2: trivial ===")
T2 = ["#####", "#SG##", "#####"]
r = run(T2, astar); valid(T2, r); show(r); assert r["length"] == 1

print("\n=== Test 3: no solution ===")
T3 = ["#######", "#S....#", "###.###", "#...#G#", "#######"]
for name, fn in (("A*", astar), ("BFS", bfs)):
    r = run(T3, fn); print(name, end=": "); show(r); assert not r["found"]

print("\n=== Test 4: alternative paths (open 5x5, S and G at opposite corners) ===")
T4 = ["#######", "#S....#", "#.....#", "#.....#", "#.....#", "#....G#", "#######"]
ra, rb = run(T4, astar), run(T4, bfs); valid(T4, ra)
print("A*  length", ra["length"], "expanded", ra["expanded"])
print("BFS length", rb["length"], "expanded", rb["expanded"])
assert ra["length"] == rb["length"] == 8
print("A* length equals BFS (known optimal) length 8 -> shortest path confirmed")

print("\n=== Task 5: BFS vs A* on the warehouse ===")
a, b = run(WAREHOUSE, astar), run(WAREHOUSE, bfs)
print(f"BFS: found={b['found']} length={b['length']} expanded={b['expanded']}")
print(f"A* : found={a['found']} length={a['length']} expanded={a['expanded']}")
print("same path:", a["path"] == b["path"])
n_free = sum(ch != "#" for row in WAREHOUSE for ch in row)
print("free cells (incl S,G):", n_free)
print("(also on open 7x7 map T4) BFS expanded", rb["expanded"], " A* expanded", ra["expanded"])

print("\n=== Task 6: heuristic variants ===")
OPEN = ["#" * 12] + ["#S" + "." * 9 + "#"] + ["#" + "." * 10 + "#"] * 4 + ["#" + "." * 9 + "G#"] + ["#" * 12]
for title, lay in (("warehouse", WAREHOUSE), ("open 6x10 room with no walls (extra)", OPEN)):
    print("--", title)
    for name, h in (("Manhattan", manhattan), ("h=0", zero), ("Euclidean", euclidean), ("2*Manhattan", double_manhattan)):
        r = run(lay, astar, h=h)
        print(f"{name:12s} found={r['found']} length={r['length']} expanded={r['expanded']}")

print("\n=== A* tie-breaking (prefer lower h among equal f) ===")
for title, lay in (("warehouse", WAREHOUSE), ("open 6x10 room", OPEN), ("open 5x5 room (Test 4)", T4)):
    a0, a1 = run(lay, astar), run(lay, astar, prefer_low_h=True)
    print(f"{title:24s} default tie-break expanded={a0['expanded']} length={a0['length']} | prefer low h expanded={a1['expanded']} length={a1['length']}")

print("\n=== over-optimistic vs over-aggressive heuristics: a map where an inflated h gives a longer path ===")
T6 = ["#########", "#S......#", "#.#.....#", "#.#...#.#", "#.##..#.#", "#.....#G#", "#########"]
g, s, t = parse(T6)
print("BFS (optimal) length:", bfs(g, s, t)["length"])
for k in (0, 1, 2, 3, 10):
    r = astar(g, s, t, h=lambda a, b, k=k: k * manhattan(a, b))
    print(f"h = {k}*Manhattan: length={r['length']} expanded={r['expanded']}")
