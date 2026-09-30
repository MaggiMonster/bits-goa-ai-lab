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
