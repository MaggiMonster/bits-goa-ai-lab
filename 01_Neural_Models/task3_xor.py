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
