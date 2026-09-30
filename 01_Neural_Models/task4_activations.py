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
