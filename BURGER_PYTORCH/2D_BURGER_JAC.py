import time
import torch
from torch.func import jacrev

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

dimX = 21
dimY = 21
dimT = 201
dx = 0.05
dy = 0.05
dt = 1e-1
x = torch.arange(0, 1+dx, dx)
y = torch.arange(0, 1+dy, dx)
t = torch.arange(0, 2+dt, dt)
u = torch.zeros(dimT, dimX, dimY, device=device)
v = torch.zeros(dimT, dimX, dimY, device=device)
R = 100

def uC(x, y, t):
    X, Y = torch.meshgrid(x, y, indexing='xy')
    return 0.75 - (4*(1+torch.exp((-4*X + 4*Y - t)*R/32)))**(-1)
def vC(x, y, t):
    X, Y = torch.meshgrid(x, y, indexing='xy')
    return 0.75 + (4*(1+torch.exp((-4*X + 4*Y - t)*R/32)))**(-1)

u[0] = uC(x, y, t = 0)
v[0] = vC(x, y, t = 0)
for i in range(len(t)):
    uCi = uC(x, y, t[i])
    vCi = vC(x, y, t[i])
    u[:, :, 0][i] = uCi[:, 0]
    v[:, :, 0][i] = vCi[:, 0]
    u[:, :, -1][i] = uCi[:, -1]
    v[:, :, -1][i] = vCi[:, -1]
    u[:, 0, :][i] = uCi[0, :]
    v[:, 0, :][i] = vCi[0, :]
    u[:, -1, :][i] = uCi[-1, :]
    v[:, -1, :][i] = vCi[-1, :]

n= 0
def f(X):
    X = X.view(2*(dimY-2), (dimX-2))
    N = (dimY-2)
    F1 = torch.zeros(2*((dimY-2)*(dimX-2)), )
    F1 = F1.to(device=device)
    pos = 0
    for j in range((dimY)):
        for i in range((dimX)):
            if j == 1:
                if i == 1:
                    F1[pos] = (1/dt)*(X[j-1, i-1]   - u[n][j, i]) + (1/(2*dx))*(X[j-1, i] - u[:, :, 0][n+1][j])*X[j-1, i-1]    +   (1/(2*dy))*(X[j, i-1] - u[:, 0, :][n+1][i])*X[N+j-1, i-1]    -    (1/(R*dx**2))*(X[j-1, i] - 2*X[j-1, i-1] + u[:, :, 0][n+1][j])    -    (1/(R*dy**2))*(X[j, i-1] - 2*X[j-1, i-1] + u[:, 0, :][n+1][i])
                    
                    F1[(dimY-2)*(dimX-2)+pos] = (1/dt)*(X[N+j-1, i-1]   - v[n][j, i]) + (1/(2*dx))*(X[N+j-1, i] - v[:, :, 0][n+1][j])*X[j-1, i-1]    +   (1/(2*dy))*(X[N+j, i-1] - v[:, 0, :][n+1][i])*X[N+j-1, i-1]    -    (1/(R*dx**2))*(X[N+j-1, i] - 2*X[N+j-1, i-1] + v[:, :, 0][n+1][j])    -    (1/(R*dy**2))*(X[N+j, i-1] - 2*X[N+j-1, i-1] + v[:, 0, :][n+1][i])
                    
                    pos +=1                   
                elif i == dimX-2:           
                    F1[pos] = (1/dt)*(X[j-1, i-1]   - u[n][j, i]) + (1/(2*dx))*(u[:, :, -1][n+1][j] - X[j-1, i-2])*X[j-1, i-1] +   (1/(2*dy))*(X[j, i-1] - u[:, 0, :][n+1][i])*X[N+j-1, i-1]    -    (1/(R*dx**2))*(u[:, :, -1][n+1][j] - 2*X[j-1, i-1] + X[j-1, i-2]) -    (1/(R*dy**2))*(X[j, i-1] - 2*X[j-1, i-1] + u[:, 0, :][n+1][i])
                    
                    F1[(dimY-2)*(dimX-2)+pos] = (1/dt)*(X[N+j-1, i-1]   - v[n][j, i]) + (1/(2*dx))*(v[:, :, -1][n+1][j] - X[N+j-1, i-2])*X[j-1, i-1] +   (1/(2*dy))*(X[N+j, i-1] - v[:, 0, :][n+1][i])*X[N+j-1, i-1]    -    (1/(R*dx**2))*(v[:, :, -1][n+1][j] - 2*X[N+j-1, i-1] + X[N+j-1, i-2]) -    (1/(R*dy**2))*(X[N+j, i-1] - 2*X[N+j-1, i-1] + v[:, 0, :][n+1][i])
                     
                    pos +=1                  
                elif i>1 and i <dimX-2:           
                    F1[pos] = (1/dt)*(X[j-1, i-1]   - u[n][j, i]) + (1/(2*dx))*(X[j-1, i] - X[j-1, i-2])*X[j-1, i-1]        +   (1/(2*dy))*(X[j, i-1] - u[:, 0, :][n+1][i])*X[N+j-1, i-1]    -    (1/(R*dx**2))*(X[j-1, i] - 2*X[j-1, i-1] + X[j-1, i-2])        -    (1/(R*dy**2))*(X[j, i-1] - 2*X[j-1, i-1] + u[:, 0, :][n+1][i])
                    
                    F1[(dimY-2)*(dimX-2)+pos] = (1/dt)*(X[N+j-1, i-1]   - v[n][j, i]) + (1/(2*dx))*(X[N+j-1, i] - X[N+j-1, i-2])*X[j-1, i-1]        +   (1/(2*dy))*(X[N+j, i-1] - v[:, 0, :][n+1][i])*X[N+j-1, i-1]    -    (1/(R*dx**2))*(X[N+j-1, i] - 2*X[N+j-1, i-1] + X[N+j-1, i-2])        -    (1/(R*dy**2))*(X[N+j, i-1] - 2*X[N+j-1, i-1] + v[:, 0, :][n+1][i])
                    
                    pos +=1   

            elif j == dimY-2:           
                if i == 1:           
                    F1[pos] = (1/dt)*(X[j-1, i-1]   - u[n][j, i]) + (1/(2*dx))*(X[j-1, i] - u[:, :, 0][n+1][j])*X[j-1, i-1]    +   (1/(2*dy))*(u[:, -1, :][n+1][i] - X[j-2, i-1])*X[N+j-1, i-1] -    (1/(R*dx**2))*(X[j-1, i] - 2*X[j-1, i-1] + u[:, :, 0][n+1][j])    -    (1/(R*dy**2))*(u[:, -1, :][n+1][i] - 2*X[j-1, i-1] + X[j-2, i-1])
                
                    F1[(dimY-2)*(dimX-2)+pos] = (1/dt)*(X[N+j-1, i-1]   - v[n][j, i]) + (1/(2*dx))*(X[N+j-1, i] - v[:, :, 0][n+1][j])*X[j-1, i-1]    +   (1/(2*dy))*(v[:, -1, :][n+1][i] - X[N+j-2, i-1])*X[N+j-1, i-1] -    (1/(R*dx**2))*(X[N+j-1, i] - 2*X[N+j-1, i-1] + v[:, :, 0][n+1][j])    -    (1/(R*dy**2))*(v[:, -1, :][n+1][i] - 2*X[N+j-1, i-1] + X[N+j-2, i-1])
                    
                    pos +=1                   
                elif i == dimX-2:           
                    F1[pos] = (1/dt)*(X[j-1, i-1]   - u[n][j, i]) + (1/(2*dx))*(u[:, :, -1][n+1][j] - X[j-1, i-2])*X[j-1, i-1] +   (1/(2*dy))*(u[:, -1, :][n+1][i] - X[j-2, i-1])*X[N+j-1, i-1] -    (1/(R*dx**2))*(u[:, :, -1][n+1][j] - 2*X[j-1, i-1] + X[j-1, i-2]) -    (1/(R*dy**2))*(u[:, -1, :][n+1][i] - 2*X[j-1, i-1] + X[j-2, i-1])
                
                    F1[(dimY-2)*(dimX-2)+pos] = (1/dt)*(X[N+j-1, i-1]   - v[n][j, i]) + (1/(2*dx))*(v[:, :, -1][n+1][j] - X[N+j-1, i-2])*X[j-1, i-1] +   (1/(2*dy))*(v[:, -1, :][n+1][i] - X[N+j-2, i-1])*X[N+j-1, i-1] -    (1/(R*dx**2))*(v[:, :, -1][n+1][j] - 2*X[N+j-1, i-1] + X[N+j-1, i-2]) -    (1/(R*dy**2))*(v[:, -1, :][n+1][i] - 2*X[N+j-1, i-1] + X[N+j-2, i-1])
                    
                    pos +=1      
                elif i>1 and i <dimX-2:           
                    F1[pos] = (1/dt)*(X[j-1, i-1]   - u[n][j, i]) + (1/(2*dx))*(X[j-1, i] - X[j-1, i-2])*X[j-1, i-1]        +   (1/(2*dy))*(u[:, -1, :][n+1][i] - X[j-2, i-1])*X[N+j-1, i-1] -    (1/(R*dx**2))*(X[j-1, i] - 2*X[j-1, i-1] + X[j-1, i-2])        -    (1/(R*dy**2))*(u[:, -1, :][n+1][i] - 2*X[j-1, i-1] + X[j-2, i-1])               

                    F1[(dimY-2)*(dimX-2)+pos] = (1/dt)*(X[N+j-1, i-1]   - v[n][j, i]) + (1/(2*dx))*(X[N+j-1, i] - X[N+j-1, i-2])*X[j-1, i-1]        +   (1/(2*dy))*(v[:, -1, :][n+1][i] - X[N+j-2, i-1])*X[N+j-1, i-1] -    (1/(R*dx**2))*(X[N+j-1, i] - 2*X[N+j-1, i-1] + X[N+j-1, i-2])        -    (1/(R*dy**2))*(v[:, -1, :][n+1][i] - 2*X[N+j-1, i-1] + X[N+j-2, i-1])               
                    
                    pos +=1

            elif j >1 and j <dimY-2:           
                if i == 1:           
                    F1[pos] = (1/dt)*(X[j-1, i-1]   - u[n][j, i]) + (1/(2*dx))*(X[j-1, i] - u[:, :, 0][n+1][j])*X[j-1, i-1]    +   (1/(2*dy))*(X[j, i-1] - X[j-2, i-1])*X[N+j-1, i-1]        -    (1/(R*dx**2))*(X[j-1, i] - 2*X[j-1, i-1] + u[:, :, 0][n+1][j])    -    (1/(R*dy**2))*(X[j, i-1] - 2*X[j-1, i-1] + X[j-2, i-1])
                
                    F1[(dimY-2)*(dimX-2)+pos] = (1/dt)*(X[N+j-1, i-1]   - v[n][j, i]) + (1/(2*dx))*(X[N+j-1, i] - v[:, :, 0][n+1][j])*X[j-1, i-1]    +   (1/(2*dy))*(X[N+j, i-1] - X[N+j-2, i-1])*X[N+j-1, i-1]        -    (1/(R*dx**2))*(X[N+j-1, i] - 2*X[N+j-1, i-1] + v[:, :, 0][n+1][j])    -    (1/(R*dy**2))*(X[N+j, i-1] - 2*X[N+j-1, i-1] + X[N+j-2, i-1])
                    
                    pos +=1   
                elif i == dimX-2:           
                    F1[pos] = (1/dt)*(X[j-1, i-1]   - u[n][j, i]) + (1/(2*dx))*(u[:, :, -1][n+1][j] - X[j-1, i-2])*X[j-1, i-1] +   (1/(2*dy))*(X[j, i-1] - X[j-2, i-1])*X[N+j-1, i-1]        -    (1/(R*dx**2))*(u[:, :, -1][n+1][j] - 2*X[j-1, i-1] + X[j-1, i-2]) -    (1/(R*dy**2))*(X[j, i-1] - 2*X[j-1, i-1] + X[j-2, i-1])
                
                    F1[(dimY-2)*(dimX-2)+pos] = (1/dt)*(X[N+j-1, i-1]   - v[n][j, i]) + (1/(2*dx))*(v[:, :, -1][n+1][j] - X[N+j-1, i-2])*X[j-1, i-1] +   (1/(2*dy))*(X[N+j, i-1] - X[N+j-2, i-1])*X[N+j-1, i-1]        -    (1/(R*dx**2))*(v[:, :, -1][n+1][j] - 2*X[N+j-1, i-1] + X[N+j-1, i-2]) -    (1/(R*dy**2))*(X[N+j, i-1] - 2*X[N+j-1, i-1] + X[N+j-2, i-1])
                    
                    pos +=1   
                elif i>1 and i <dimX-2:           
                    F1[pos] = (1/dt)*(X[j-1, i-1]   - u[n][j, i]) + (1/(2*dx))*(X[j-1, i] - X[j-1, i-2])*X[j-1, i-1]        +   (1/(2*dy))*(X[j, i-1] - X[j-2, i-1])*X[N+j-1, i-1]        -    (1/(R*dx**2))*(X[j-1, i] - 2*X[j-1, i-1] + X[j-1, i-2])        -    (1/(R*dy**2))*(X[j, i-1] - 2*X[j-1, i-1] + X[j-2, i-1])               

                    F1[(dimY-2)*(dimX-2)+pos] = (1/dt)*(X[N+j-1, i-1]   - v[n][j, i]) + (1/(2*dx))*(X[N+j-1, i] - X[N+j-1, i-2])*X[j-1, i-1]        +   (1/(2*dy))*(X[N+j, i-1] - X[N+j-2, i-1])*X[N+j-1, i-1]        -    (1/(R*dx**2))*(X[N+j-1, i] - 2*X[N+j-1, i-1] + X[N+j-1, i-2])        -    (1/(R*dy**2))*(X[N+j, i-1] - 2*X[N+j-1, i-1] + X[N+j-2, i-1])               
                
                    pos +=1
    return F1

X_rand = torch.rand(2*(dimY-2)*(dimX-2), )
d = 2*(dimY-2)*(dimX-2) 
t0 = time.time()
for n in range(50):
    X0 = X_rand
    X0 = X0.to(device=device)
    tol = torch.ones(d,)*9e-6
    tol = tol.to(device=device)
    e = 1
    iter = 0
    t1 = time.time()
    err = [0]
    diff = []
    while sum(abs(e)>tol) != 0:
        J = jacrev(f, argnums=0)(X0)
        J = J.to(device=device)
        F_X0 = f(X0)
        F_X0 = F_X0.to(device=device)
        X1 = X0 - torch.matmul(torch.linalg.inv(J),F_X0)
        X0 = X1
        e = F_X0
        err.append(torch.mean(abs(e)))
        diff.append(abs(err[-2] - err[-1]))
    #    if (abs(err[-2] - err[-1]) <= 1e-8):
    #        break
    #    elif iter>=10:
    #        if abs(diff[-2] - diff[-1]) <= 1e-8:
    #            break
        print(torch.mean(abs(f(X0))))
        iter += 1
    u[n+1][1:-1, 1:-1] = X0.view(2, (dimY-2), (dimX-2))[0]
    v[n+1][1:-1, 1:-1] = X0.view(2, (dimY-2), (dimX-2))[1]

    print("t = {} | iteration time = ".format(n+1), time.time() - t1, "Average Error = ".format(n+1), torch.mean(abs(f(X0))))

print("total time = {}".format(time.time() - t0))

import matplotlib.pyplot as plt
for i in range(1, len(t)):
    U = uC(x, y, t = t[i])
    V = vC(x, y, t = t[i])
    fig, axs = plt.subplots(1, 2)
    ax1 = axs[0].imshow(u[i].cpu().numpy() -U.cpu().numpy(), cmap = 'turbo')
    axs[0].set_title('ERROR u t = {}'.format(i))
    fig.colorbar(ax1)
    ax2 = axs[1].imshow(v[i].cpu().numpy() -V.cpu().numpy(), cmap = 'turbo')
    axs[1].set_title('ERROR v t = {}'.format(i))
    fig.colorbar(ax2)
    plt.savefig('t = {}.png'.format(i))