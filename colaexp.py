import numpy as np
import matplotlib.pyplot as plt
from cola import Auto, CG, eigmax
# import cola


# Function to generate the spiral dataset
def generate_spiral_data(n_points, n_spirals, noise=0.02):
    X = np.zeros((n_points * n_spirals, 2))
    y = np.zeros(n_points * n_spirals, dtype=int)
    delta_theta = 4 * np.pi / n_spirals
    for i in range(n_spirals):
        theta = np.linspace(i * delta_theta, (i + 1) * delta_theta, n_points)
        r = np.linspace(0.0, 1, n_points)
        t = theta + np.random.randn(n_points) * noise
        X[i * n_points:(i + 1) * n_points] = np.column_stack((r * np.sin(t), r * np.cos(t)))
        y[i * n_points:(i + 1) * n_points] = i
    return X, y

# Generating the spiral dataset with 100 points and 5 spirals
n_points = 200
n_spirals = 8
print('generating data')
X, y = generate_spiral_data(n_points, n_spirals)
print('data generated')

import torch
import torch.nn as nn
import torch.optim as optim

def MLP(k=200):
    return nn.Sequential(
        nn.Linear(2, k), nn.SiLU(),
        nn.Linear(k, k), nn.SiLU(),
        nn.Linear(k, n_spirals))

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)





learning_rate = 1e-2
epochs = 500

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
adam_losses = []
for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    adam_losses.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

import torch.func as tf

def flatten_params(params):
    shapes = [p.shape for p in params]
    flat_params = torch.cat([p.flatten() for p in params])
    return flat_params, shapes


def unflatten_params(flat_params, shapes):
    params = []
    i = 0
    for shape in shapes:
        size = torch.prod(torch.tensor(shape)).item()
        param = flat_params[i:i + size]
        param = param.view(shape)
        params.append(param)
        i += size
    return params

model = MLP()
flat_p, shape = flatten_params(list(model.parameters()))
flat_p = flat_p.detach().requires_grad_(True)

def stateless_model(flatparams, x):
    params = unflatten_params(flatparams, shape)
    names = list(n for n, _ in model.named_parameters())
    nps = {n: p for n, p in zip(names, params)}
    return tf.functional_call(model, nps, x)




def flat_fn(p):
    return stateless_model(p, X).reshape(-1)

def GN(p):
    """Gauss-Newton approximation to the Hessian"""
    J = cola.ops.Jacobian(flat_fn, p)
    loss = lambda z: criterion(z.reshape(X.shape[0],-1),y)*n_spirals
    H = cola.ops.Hessian(loss, flat_fn(p))
    G = J.T @ H @ J
    return cola.PSD(G+1e-3*cola.ops.I_like(G))

def Fisher(p):
    F = cola.ops.FIM(lambda p: stateless_model(p, X), p)
    return cola.PSD(F+1e-3*cola.ops.I_like(F))

def flat_loss(params):
    return criterion(flat_fn(params).reshape(X.shape[0],-1), y)

with torch.no_grad():
    print(f"GN eigmax: {cola.eigmax(GN(flat_p), alg=Auto(tol=1e-2))}")
    print(f"Fisher eigmax: {cola.eigmax(Fisher(flat_p), alg=Auto(tol=1e-2))}")


# Method 1: Hessian weighted Gauss Newton
p = flat_p.clone()

gnh_losses = []
print('gnh')
for epoch in range(epochs//5):
    with torch.no_grad(): # don't pay extra memory for recording the computation graph
        g = torch.func.grad(flat_loss)(p)
        p -= cola.inv(GN(p),alg=Auto(tol=1e-3, max_iters=20)) @ g
        loss = flat_loss(p)
        gnh_losses.append(loss.item())
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

# Method 2: Natural gradient descent
p = flat_p.clone()

print('natural gradient')
natural_gradient_losses = []
for epoch in range(epochs//5):
    with torch.no_grad(): # don't pay extra memory for recording the computation graph
        g = torch.func.grad(flat_loss)(p)
        p -= .5*cola.inv(Fisher(p),alg=Auto(tol=1e-3, max_iters=20)) @ g

        loss = flat_loss(p)
        natural_gradient_losses.append(loss.item())
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

plt.plot(adam_losses, label='Adam', color='brown')
plt.plot(natural_gradient_losses, label='Natural Gradient Descent', color='darkgreen')
plt.plot(gnh_losses, label='Gauss-Newton Method', color='darkblue')

plt.xlabel('iterations')
plt.ylabel('loss')
plt.yscale('log')
plt.legend()
plt.savefig('ngnvsadam.pdf')