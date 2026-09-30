# 01: Neural Models (XOR, depth, activations, output layers)
Files: `task3_xor.py` (2-2-1 tanh net, BCEWithLogits, Adam), `task4_diagnostics.py` (seeds, finite-difference
gradient check, zero-init symmetry), `task4_activations.py` (sigmoid/tanh/ReLU), `task5_multiclass.py`
(3-class softmax), `lab_all.py` (all four in one file), `report.pdf`, `ANSWERS.md`.

Run: `pip install -r ../requirements.txt && python task3_xor.py`

| Activation (seed 0) | Final loss | 4/4 | Early grad norm | Seeds OK /10 |
|---|---|---|---|---|
| Sigmoid | 0.4774 | no | 0.000891 | 4 |
| Tanh | 0.000083 | yes | 0.061785 | 4 |
| ReLU | 0.6931 | no | 0.001699 | 3 |

Also: autograd matches finite differences (max diff 3.9e-4, float32); zero init keeps both hidden rows identical forever.
