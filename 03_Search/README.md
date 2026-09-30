# 03: Search and A\*
`search_agent.py` has A\* (heap + closed set, optional heuristic tie-breaking) and BFS;
`tests_and_experiments.py` runs Tests 1-4, the BFS/A\* comparison and heuristic variants.

Run: `python tests_and_experiments.py`

| Measure | BFS | A\* |
|---|---|---|
| Found | yes | yes |
| Path length | 40 | 40 |
| States expanded | 63 | 63 |

The lab map is a winding maze with one route, so A\* cannot prune; on an open room A\* (low-h tie-break) expands 14 states vs 59.
On a small extra map, 3x Manhattan returns length 16 vs optimal 10. See `ANSWERS.md`, `report.pdf`.
