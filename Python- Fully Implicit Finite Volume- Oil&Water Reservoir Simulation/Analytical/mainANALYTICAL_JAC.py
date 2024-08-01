import torch
import numpy as np
from  pyResSimAnalytical import *
from torch.func import jacrev
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

c1 = 6.2283e-3
c2 = 1.0
c3 = 5.5146
C  = torch.tensor([c1, c2, c3], dtype=torch.float).to(device = device)
Dict = {}
Dict["dx"] = torch.tensor(30.0).to(device = device)
Dict["dy"] = torch.tensor(30.0).to(device = device)
Dict["dz"] = torch.tensor(30.0).to(device = device)
Dict["dt"] = 1
Dict["dV"] = Dict["dx"]*Dict["dy"]*Dict["dz"]
Dict["K"] = torch.tensor(np.genfromtxt(r'C:\Users\YUSIFOH\NNs\2D_2PHASE_FLOW_JAC\permx.dat', dtype=None, delimiter=','), device=device)
Dict["phi"] = torch.tensor(0.2).to(device = device)
Dict["Pref"] = torch.tensor(2800.0).to(device = device)
Dict["rho_o_std"] = torch.tensor(45.0).to(device = device) #lb/ft^3
Dict["rho_w_std"] = torch.tensor(62.4).to(device = device) #lb/ft^3
Dict["Co"] = torch.tensor(1e-5).to(device = device)
Dict["Cw"] = torch.tensor(3e-6).to(device = device)
Dict["Cr"] = torch.tensor(3e-6).to(device = device)
Dict["Swmin"] = torch.tensor(0.1).to(device = device)
Dict["Somin"] = torch.tensor(0.2).to(device = device)
Dict["BHP"] = torch.tensor(2900.0).to(device = device)
Dict["rw"] = torch.tensor(0.35).to(device = device)
Dict["re"] = 0.14*torch.sqrt(Dict["dx"]**2 + Dict["dy"]**2)
Dict["h"] =  torch.tensor(30.0).to(device = device)
Dict["S"] = torch.tensor(0.0).to(device = device)
Dict["QinjW"] = torch.tensor(300.0).to(device = device)
Dict["Kro_"] = torch.tensor(0.7).to(device = device)
Dict["Krw_"] = torch.tensor(0.08).to(device = device)

Days = 400
Nx, Ny, Nz = int(450/Dict["dx"]), int(450/Dict["dy"]), int(30/Dict["dz"])
N = Nx*Ny*Nz

cell_connections = np.genfromtxt(r'C:\Users\YUSIFOH\NNs\2D_2PHASE_FLOW_JAC\edge_array.dat', dtype=np.integer , encoding="UTF-8")
Well_Loc = torch.zeros(2, N, dtype=torch.float).to(device = device)
well_1 = 0   # Injector
well_2 = 224 # Producer
Well_Loc[0][well_1] = torch.tensor(-1, dtype=torch.float)
Well_Loc[1][well_2] = torch.tensor(1, dtype=torch.float)

steps = int(1 + Days/Dict["dt"])

Pi = torch.tensor(3000.0).to(device = device)
Swi = 0.2
Soi = 0.8
P_hist = torch.zeros(steps, N).to(device = device)
P_hist[0] = torch.ones(N, device=device)*Pi
Sw_hist = torch.zeros(steps, N).to(device = device)
Sw_hist[0] = torch.ones(N, device=device)*Swi

Residual_Jacobian = Residual_Jacobian(Dict, Well_Loc[1], Well_Loc[0], cell_connections)


X = torch.tensor([3000.0+i*10 for i in range(15)]*15 + [0.2+i*0.01 for i in range(15)]*15, dtype=torch.float).to(device=device)

n = 0
CondIter = 5
tol = torch.ones(2*Nx*Ny,)*1e-3
tol = tol.to(device=device)

LR = []
LJR = []
solver = []
import time
JR, R = Residual_Jacobian.calculate_Residual_Jacobian(X, C, P_hist, Sw_hist, n, N)
for n in range(0, steps-1):
    iter = 0
    e = 1
    if n != 0:
        CondIter = 3
    lst = []
    while sum(e>tol) != 0 and iter <= CondIter:
        t0 = time.time()
        JR, R = Residual_Jacobian.calculate_Residual_Jacobian(X, C, P_hist, Sw_hist, n, N)
        t1 = time.time()
        X = X - torch.linalg.matmul(torch.linalg.inv(JR), R)
        t2 = time.time()
        LR.append((t1-t0)*0.375)
        LJR.append((t1-t0)*0.625)
        solver.append(t2-t1)
        iter += 1
        e = abs(R)
    print(torch.mean(e))
    print("t = {} Done!".format(n+1))
    P_hist[n+1] = X.view(2, Ny*Nx)[0]
    Sw_hist[n+1] = X.view(2, Ny*Nx)[1]

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
fig = plt.figure(figsize=(5, 5), dpi = 110)
for i in range(0,12):
    plt.subplot(3,4,i+1)
    plt.imshow(P_hist[int((i+1)*30/Dict["dt"])+1].cpu().numpy().reshape((Nx, Ny)), cmap = 'jet', aspect=0.5)
    cbar = plt.colorbar(orientation = 'horizontal', label = 'P | Day = {}'.format(int((i+1)*30/Dict["dt"])+1))
    cbar.ax.tick_params(labelsize=6)
fig.tight_layout(pad = 1)
#plt.savefig(r'C:\Users\YUSIFOH\NNs\2D_2PHASE_FLOW_JAC\plots1\Pmap.png', dpi = 1080)
plt.show()


fig = plt.figure(figsize=(5, 5), dpi = 110)
for i in range(0,12):
    plt.subplot(3,4,i+1)
    plt.imshow(Sw_hist[int((i+1)*30/Dict["dt"])+1].cpu().numpy().reshape((Nx, Ny)), cmap = 'jet', aspect=0.5)
    cbar = plt.colorbar(orientation = 'horizontal', label = 'Sw | Day = {}'.format(int((i+1)*30/Dict["dt"])+1))
    cbar.ax.tick_params(labelsize=6)
fig.tight_layout(pad = 1)
#plt.savefig(r'C:\Users\YUSIFOH\NNs\2D_2PHASE_FLOW_JAC\plots1\Swmap.png', dpi = 1080)
plt.show()

LR = np.array(LR)
LJR = np.array(LJR)
solver = np.array(solver)

fig = plt.figure(figsize=(5, 5), dpi = 100)
plt.plot(range(len(LJR)), LJR, color = "r")
plt.scatter(range(len(LJR)), LJR, color = "r")
plt.xlabel("Iterations")
plt.ylabel("Computation Time [s]")
plt.title("Jacobian Calculation")
plt.text(0, 0, "Total = {}".format(round(sum(LJR), 2)))
plt.grid()
plt.show()

fig = plt.figure(figsize=(5, 5), dpi = 100)
plt.plot(range(len(LJR)), LR+LJR+solver, color = "r")
plt.scatter(range(len(LJR)), LR+LJR+solver, color = "r")
plt.xlabel("Iterations")
plt.ylabel("Computation Time [s]")
plt.title("Complete Calculation")
plt.text(0, 0, "Total = {}".format(round(sum(LR+LJR+solver), 2)))
plt.grid()
plt.show()

print("TOTAL         :: ", "Residual = ",  round(sum(LR), 2), "Jacobian = ",round(sum(LJR), 2), "Solver = ", round(sum(solver), 2))
print("PER ITERATION :: " ,"Residual = ",  round(sum(LR)/len(LR), 3), "Jacobian = ",round(sum(LJR)/len(LJR), 3), "Solver = ", round(sum(solver)/len(solver), 3))

#TOTAL         ::  Residual =  4.53 Jacobian =  20.93 Solver =  4.84
#PER ITERATION ::  Residual =  0.003 Jacobian =  0.013 Solver =  0.003

#best

# TOTAL         ::  Residual =  4.25 Jacobian =  19.23 Solver =  4.63 ----> 28 seconds
# PER ITERATION ::  Residual =  0.003 Jacobian =  0.012 Solver =  0.003
