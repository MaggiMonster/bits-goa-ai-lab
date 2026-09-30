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
