import time
import torch
from torch.func import jacrev

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

d = 300
x = torch.linspace(-1, 1, d)
M = 100
dx = 2/(d - 1)
dt = 0.01
u = torch.zeros(M, d, device=device)
u[0] = -torch.sin(torch.pi*x)
c = 0.01/torch.pi

def f(X):
    F = torch.zeros(d-2,)
    F = F.to(device=device)
    for i in range(0, d-2):
        if i == 0:
            print("XXXXXXXXX", (1/dt)*(X[i] - u[n][i+1]) + (X[i]/(2*dx))*(X[i+1] - torch.tensor(0, device=device)) - (c/dx**2)*(X[i+1] -2*X[i] + torch.tensor(0, device=device)))
            F[i] = (1/dt)*(X[i] - u[n][i+1]) + (X[i]/(2*dx))*(X[i+1] - torch.tensor(0, device=device)) - (c/dx**2)*(X[i+1] -2*X[i] + torch.tensor(0, device=device))
        elif i>0 and i<d-3:
            F[i] = (1/dt)*(X[i] - u[n][i+1]) + (X[i]/(2*dx))*(X[i+1] - X[i-1]) - (c/dx**2)*(X[i+1] -2*X[i] + X[i-1])
        elif i == d-3:
            F[i] = (1/dt)*(X[i] - u[n][i+1]) + (X[i]/(2*dx))*(torch.tensor(0, device=device) - X[i-1]) - (c/dx**2)*(torch.tensor(0, device=device) -2*X[i] + X[i-1])
    return F
X_rand = torch.rand(d-2,)
t0 = time.time()
for n in range(M-1):
    X0 = X_rand
    X0 = X0.to(device=device)
    tol = torch.ones(d-2,)*9e-6
    tol = tol.to(device=device)
    e = 1
    iter = 0
    t1 = time.time()
    err = [0]
    diff = []
    while sum(abs(e)>tol) != 0 and iter <= 15:
        J = jacrev(f, argnums=0)(X0)
        J = J.to(device=device)
        F_X0 = f(X0)
        F_X0 = F_X0.to(device=device)
        X1 = X0 - torch.matmul(torch.linalg.inv(J),F_X0)
        X0 = X1
        e = F_X0
        err.append(torch.mean(abs(e)))
        diff.append(abs(err[-2] - err[-1]))
        if (abs(err[-2] - err[-1]) <= 1e-8):
            break
        elif iter>=10:
            if abs(diff[-2] - diff[-1]) <= 1e-8:
                break
        iter += 1
    u[n+1][1:-1] = X0

    print("t = {} | iteration time = ".format(n+1), time.time() - t1, "Average Error = ".format(n+1), torch.mean(abs(f(X0))))

print("total time = {}".format(time.time() - t0))


import scipy.io
import matplotlib.pyplot as plt

data = scipy.io.loadmat(r'C:\Users\YUSIFOH\NNs\BURGERS_1D_PINN-LBFGS\burgers_shock.mat')

x1 = data['x']
t1 = data['t']
u1 = data['usol']

U = u.cpu().numpy()

plt.scatter(x[1:-1], U[25][1:-1], color = 'r')
plt.plot(x1, u1[:, 25])
plt.show()

plt.scatter(x[1:-1], U[50][1:-1], color = 'r')
plt.plot(x1, u1[:, 50])
plt.show()

plt.scatter(x[1:-1], U[75][1:-1], color = 'r')
plt.plot(x1, u1[:, 75])

plt.show()

plt.scatter(x[1:-1], U[-1][1:-1], color = 'r')
plt.plot(x1, u1[:, -1])
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(U, cmap = 'clim')
plt.colorbar()
plt.show()