# 04: Logical planning
`planner.py`: STRIPS-style actions (positive/negative preconditions and effects), BFS planner, independent plan verifier.
`planner.pl`: Prolog facts/rules (`connected`, `can_move`, `valid_move`, `valid_path`, `reduce_speed`); `run_prolog.sh` runs the queries.

Run: `python planner.py` and `sh run_prolog.sh` (needs `swipl`)

Plan: PickUp(Package,A), Move(A,B), Move(B,C), Drop(Package,C). No PickUp -> "No plan found".
See `ANSWERS.md`, `report.pdf`.
