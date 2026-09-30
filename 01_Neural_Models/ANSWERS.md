# Lab 01: Written answers

## Task 1: problem and linear model
- **Spec:** X = {0,1}^2, Y = {0,1}; (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0 (XOR).
- **Why no line works:** class 0 sits at (0,0),(1,1), class 1 at (0,1),(1,0), on opposite diagonals. A line w1x1+w2x2+b=0 needs b<0, w1+w2+b<0 (class 0) and w1+b>0, w2+b>0 (class 1); adding the last two gives w1+w2+2b>0 while adding the first two gives <0, a contradiction.
- **Prediction for affine + sigmoid:** outputs about 0.5 for every input (loss near ln 2), at most 3 of 4 correct.

## Task 2: design
2-2-1, tanh hidden, sigmoid output via BCEWithLogitsLoss, Adam, full batch.
1. *Why a hidden nonlinearity:* affine layers compose into one affine map, which is a linear classifier; the nonlinearity lets the hidden layer re-represent the inputs so XOR becomes separable.
2. *Why sigmoid + BCE:* one yes/no target; sigmoid gives a Bernoulli probability, BCE is its negative log-likelihood, and the logit gradient is simply p - y.
3. *Success criteria:* final loss well below ln 2; all four labels correct; nonzero first-layer gradient that agrees with finite differences; repeated seeds; the zero-init control fails.
- *Think about it, hidden targets:* no hidden unit is given a target; the output loss sends dL/dh back through the chain rule, and that signal alone decides what each unit computes.

## Task 3 think-about-it
Verifiable from the code alone: architecture, loss/output pairing, where forward/loss/backward/step occur, the data. Needs execution: whether it learns, final loss, gradient values, seed sensitivity.

## Task 4
- **Part B:** `parameter.grad` is dL/dW1, the derivative of the scalar mean loss w.r.t. each first-layer weight. The mean-loss gradient equals the average of the four per-example gradients because differentiation is linear (verified numerically).
- **Part C:** the two hidden rows stay identical at every step (they stay exactly 0): same inputs, same outputs, same gradient, so the same update, forever. Loss stays at ln 2.
- **Part D:** tanh learned XOR for seed 0, sigmoid and ReLU did not, but over seeds 0-9 the successes are 4, 4 and 3 of 10, so no activation is claimed "best". Sigmoid ended saturated (pre-activations around +/-11 to +/-25, activations near 0/1); ReLU ended dead (all pre-activations negative, all activations exactly 0). Distinguish them by checking pre-activations: large magnitude with saturated output (sigmoid) versus negative with output exactly 0 (ReLU).

## Task 5
Weight matrix 3x2; 3 logits per example; softmax sums to 1 by construction; the logit gradient is p - y because L = -log p_y so dL/dz_k = p_k - 1[k=y] (divided by N for a mean loss, verified). Adding a constant to all logits cancels in softmax; stable code subtracts the max logit to avoid exp overflow (naive +1000 gives NaN).
- *Think about it, scale to a large vocabulary:* softmax + cross-entropy and the p - y gradient stay the same; what changes is the surrounding architecture (embeddings, deep context-mixing layers, cost of the output layer, sampling).

## Reflection questions
1. Depth without nonlinearity adds nothing; nonlinearity is what makes XOR representable.
2. The loss fell from 0.715 to 8e-5, all four labels flipped to correct, and autograd matched finite differences: a useful signal, not just a nonzero one.
3. Identical hidden units get identical gradients, so symmetry is never broken; random init breaks it.
4. Scientific: sigmoid derivative <= 0.25 and can saturate, tanh up to 1, ReLU 1 when active else 0. Engineering: early gradient norms 0.00089 / 0.0618 / 0.0017 for seed 0, yet success rates over 10 seeds were similar.
5. The output activation and loss must match the target's probabilistic model (Bernoulli: sigmoid+BCE; one-of-K: softmax+CE) to get correct likelihoods and the clean p - y gradient.
6. LLM productivity: boilerplate, diagnostics and tests written quickly. Human verification essential: confirming the code is the specified 2-2-1 experiment, noticing seed 1 and ReLU fail, reading the finite-difference gap as float32 roundoff.
7. Keep: loss/prediction checks, gradient-norm monitoring, NaN checks, multi-seed runs. Too expensive at scale: exhaustive finite-difference checks (cost proportional to parameter count).
