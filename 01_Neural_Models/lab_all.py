"""Combined lab code: binary XOR (task3), diagnostics (task4 A-C), activations (task4 D), 3-class (task5)."""

# ================= task3_xor.py =================
print()
print('##### task3_xor')
import torch
import torch.nn as nn

torch.manual_seed(0)

# XOR dataset (sensor-disagreement warning), full batch
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# 2 -> 2 (tanh) -> 1 logit, default PyTorch init
layer1 = nn.Linear(2, 2)
layer2 = nn.Linear(2, 1)
params = list(layer1.parameters()) + list(layer2.parameters())

criterion = nn.BCEWithLogitsLoss()  # mean reduction
optimizer = torch.optim.Adam(params, lr=0.05)


def forward(inp):
    return layer2(torch.tanh(layer1(inp)))  # forward pass -> logits


with torch.no_grad():
    initial_loss = criterion(forward(x), y).item()

for step in range(3000):
    optimizer.zero_grad()
    logits = forward(x)            # forward pass
    loss = criterion(logits, y)    # scalar loss formed here
    loss.backward()                # reverse-mode AD
    optimizer.step()               # parameters updated here
    assert torch.isfinite(loss), f"loss not finite at step {step}"   # change 2: NaN/inf guard
    if step % 500 == 0:                                              # change 1: log loss every 500 steps
        print(f"step {step:4d}  loss {loss.item():.6f}")

# Gradient of first-layer weight after a fresh forward + backward
optimizer.zero_grad()
final_loss_t = criterion(forward(x), y)
final_loss_t.backward()
grad_w1 = layer1.weight.grad.clone()

with torch.no_grad():
    probs = torch.sigmoid(forward(x))
    labels = (probs >= 0.5).float()
    correct = bool((labels == y).all())

print(f"Initial loss: {initial_loss:.6f}")            # test 1: loss started high (~ln2)
print(f"Final loss:   {final_loss_t.item():.6f}")     # test 2: loss decreased -> learning
print("Probabilities:", probs.squeeze().tolist())     # test 3: outputs near 0/1
print("Labels (>=0.5):", labels.squeeze().tolist())   # test 4: thresholded predictions
print("All 4 correct:", correct)                      # test 5: matches XOR targets
print("layer1.weight.grad =\n", grad_w1)              # test 6: gradient tensor dL/dW1

# ================= task4_diagnostics.py =================
print()
print('##### task4_diagnostics')
import torch
import torch.nn as nn

x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])
crit = nn.BCEWithLogitsLoss()


def make(seed=None, zero=False):
    if seed is not None:
        torch.manual_seed(seed)
    l1, l2 = nn.Linear(2, 2), nn.Linear(2, 1)
    if zero:
        for p in list(l1.parameters()) + list(l2.parameters()):
            nn.init.zeros_(p)
    return l1, l2


def fwd(l1, l2, inp):
    return l2(torch.tanh(l1(inp)))


def train(l1, l2, steps=3000, record=()):
    opt = torch.optim.Adam(list(l1.parameters()) + list(l2.parameters()), lr=0.05)
    snaps = {}
    if 0 in record:
        snaps[0] = l1.weight.detach().clone()
    for s in range(1, steps + 1):
        opt.zero_grad()
        crit(fwd(l1, l2, x), y).backward()
        opt.step()
        if s in record:
            snaps[s] = l1.weight.detach().clone()
    return snaps


def evaluate(l1, l2):
    with torch.no_grad():
        loss = crit(fwd(l1, l2, x), y).item()
        p = torch.sigmoid(fwd(l1, l2, x))
        ok = bool(((p >= 0.5).float() == y).all())
    return loss, p.squeeze().tolist(), ok


print("=== Part A: seeds 0-4 ===")
for seed in range(5):
    l1, l2 = make(seed)
    with torch.no_grad():
        init = crit(fwd(l1, l2, x), y).item()
    train(l1, l2)
    fl, _, ok = evaluate(l1, l2)
    print(f"seed {seed}: initial {init:.4f}  final {fl:.6f}  4/4 correct: {ok}")

print("\n=== Part B: gradient check (seed 0, at init) ===")
l1, l2 = make(0)
crit(fwd(l1, l2, x), y).backward()
g = l1.weight.grad.clone()
print("autograd layer1.weight.grad =\n", g)

eps = 1e-4
fd = torch.zeros_like(g)
with torch.no_grad():
    for i in range(2):
        for j in range(2):
            orig = l1.weight[i, j].item()
            l1.weight[i, j] = orig + eps
            lp = crit(fwd(l1, l2, x), y).item()
            l1.weight[i, j] = orig - eps
            lm = crit(fwd(l1, l2, x), y).item()
            l1.weight[i, j] = orig
            fd[i, j] = (lp - lm) / (2 * eps)
print("finite-difference =\n", fd)
print("max abs diff:", (g - fd).abs().max().item())

# mean-loss gradient == average of per-example gradients
per = []
for k in range(4):
    l1.zero_grad(); l2.zero_grad()
    crit(fwd(l1, l2, x[k:k + 1]), y[k:k + 1]).backward()
    per.append(l1.weight.grad.clone())
avg = torch.stack(per).mean(0)
print("average of 4 per-example grads =\n", avg)
print("equals mean-loss grad:", torch.allclose(avg, g, atol=1e-6))

print("\n=== Part C: all-zero initialisation ===")
l1, l2 = make(zero=True)
rec = (0, 1, 2, 5, 10, 100, 3000)
snaps = train(l1, l2, record=rec)
for s in rec:
    W = snaps[s]
    print(f"step {s:4d}: row0={W[0].tolist()} row1={W[1].tolist()} identical={torch.equal(W[0], W[1])}")
fl, p, ok = evaluate(l1, l2)
print(f"final loss {fl:.6f}  probs {p}  4/4 correct: {ok}")

# ================= task4_activations.py =================
print()
print('##### task4_activations')
import torch
import torch.nn as nn

x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])
crit = nn.BCEWithLogitsLoss()
acts = {"Sigmoid": torch.sigmoid, "Tanh": torch.tanh, "ReLU": torch.relu}


def run(act, seed):
    torch.manual_seed(seed)
    l1, l2 = nn.Linear(2, 2), nn.Linear(2, 1)
    opt = torch.optim.Adam(list(l1.parameters()) + list(l2.parameters()), lr=0.05)
    early = None
    for s in range(1, 3001):
        opt.zero_grad()
        loss = crit(l2(act(l1(x))), y)
        loss.backward()
        if s == 1:
            early = l1.weight.grad.norm().item()  # Euclidean (Frobenius) norm at step 1
        opt.step()
    with torch.no_grad():
        a = l1(x)
        h = act(a)
        fl = crit(l2(h), y).item()
        ok = bool(((torch.sigmoid(l2(h)) >= 0.5).float() == y).all())
    return fl, ok, early, a, h


print("Hidden activation | Final loss | 4/4 correct? | Early ||grad W1||_2   (seed 0)")
details = {}
for n, f in acts.items():
    fl, ok, g, a, h = run(f, 0)
    details[n] = (a, h)
    print(f"{n:<17} | {fl:.6f}   | {str(ok):<12} | {g:.6f}")

print("\nFinal hidden pre-activations / activations (seed 0):")
for n, (a, h) in details.items():
    print(f"--- {n}\npre-activation:\n{a}\nactivation:\n{h}")

print("\nSuccess counts over seeds 0-9:")
for n, f in acts.items():
    wins = sum(run(f, s)[1] for s in range(10))
    print(f"{n}: {wins}/10")

# ================= task5_multiclass.py =================
print()
print('##### task5_multiclass')
import torch
import torch.nn as nn

torch.manual_seed(0)

x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([0, 1, 1, 2])  # class indices

layer1 = nn.Linear(2, 2)
layer2 = nn.Linear(2, 3)  # 3 logits
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(list(layer1.parameters()) + list(layer2.parameters()), lr=0.05)


def fwd(inp):
    return layer2(torch.tanh(layer1(inp)))


print("Final weight matrix shape:", tuple(layer2.weight.shape))
print("Logits shape per example:", tuple(fwd(x[:1]).shape[1:]))

for _ in range(3000):
    opt.zero_grad()
    crit(fwd(x), y).backward()
    opt.step()

with torch.no_grad():
    logits = fwd(x)
    p = torch.softmax(logits, dim=1)
print("Final loss:", crit(logits, y).item())
print("Softmax probabilities:\n", p)
print("Predicted classes:", p.argmax(1).tolist(), " targets:", y.tolist())
print("Example 1 probs:", p[1].tolist(), " sum =", p[1].sum().item())

# gradient wrt logits == (p - y_onehot)/N
z = fwd(x).detach().requires_grad_(True)
crit(z, y).backward()
expected = (torch.softmax(z.detach(), 1) - nn.functional.one_hot(y, 3).float()) / 4
print("dL/dlogits == (p - y)/N:", torch.allclose(z.grad, expected, atol=1e-7))

# shift invariance and stability
print("softmax(z+100) == softmax(z):", torch.allclose(torch.softmax(logits + 100, 1), p, atol=1e-6))
big = logits[0] + 1000
naive = torch.exp(big) / torch.exp(big).sum()
stable = torch.exp(big - big.max()) / torch.exp(big - big.max()).sum()
print("naive exp with +1000:", naive.tolist())
print("max-subtracted with +1000:", stable.tolist())
