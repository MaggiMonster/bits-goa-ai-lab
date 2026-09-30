# AI Lab Work: Neural Models, Agents, Search, Logical Planning, Bayesian Networks

Five labs on core AI ideas, each implemented in plain Python (PyTorch only for lab 1), tested with
explicit checks, and written up in a PDF report. All code was developed with an LLM assistant; the
prompts used are recorded per lab (`prompts.txt` or the report appendix).

| # | Lab | Core idea | Run | Headline result |
|---|-----|-----------|-----|-----------------|
| 01 | [Neural Models](01_Neural_Models/) | XOR needs a nonlinear hidden layer; backprop, symmetry, activations, softmax | `python task3_xor.py` (or `lab_all.py`) | tanh 2-2-1 net reaches loss 8e-5; zero init stays symmetric at loss ln 2; 3-class softmax gradient = (p-y)/N |
| 02 | [Agents](02_Agents/) | Goal-based agent, BFS on a grid | `python warehouse_agent.py` | 20-move shortest path, verified against an independent distance check |
| 03 | [Search](03_Search/) | A\* vs BFS, heuristics | `python tests_and_experiments.py` | A\* = BFS (63 expansions) on the maze; 14 vs 59 on an open room; 3x Manhattan gives a longer path |
| 04 | [Logical Planning](04_Logical_Planning/) | STRIPS planner + BFS; Prolog as verifier | `python planner.py`, `sh run_prolog.sh` | 4-step plan verified; "no plan" without PickUp; Prolog rejects Move(a,c) |
| 05 | [Bayesian Networks](05_Bayesian_Networks/) | n-gram LM as a Bayesian network | `python ngram_lm.py` | all CPT rows sum to 1; 2nd-order: 88% of contexts unseen, 0 novel sentences |

## Layout of every lab folder
- `README.md`: what the lab does, how to run it, key results
- `ANSWERS.md`: the written (subjective) questions and their answers
- `report.pdf` (+ `report.tex`): the full report with prompts, results and code
- source files, `prompts.txt` and captured program output (`results*.txt`)

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # torch + numpy, needed only for lab 01
# lab 04 optional: SWI-Prolog (brew install swi-prolog)
```
Run each script from inside its lab folder. Results are seeded where randomness is used.
