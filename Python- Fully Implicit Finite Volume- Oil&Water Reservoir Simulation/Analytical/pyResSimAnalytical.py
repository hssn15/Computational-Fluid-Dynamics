import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class properties:
    # MEDIA PROPERTIES:
    def porosity(P, Cr, Pref, phi):
        return phi*torch.exp(Cr*(P - Pref))

    # OIL PROPERTIES
    def rho_o(P, Co, Pref, rho_o_std):
        return rho_o_std*torch.exp(Co*(P - Pref))

    def mu_o(P, Co, Pref):
        return 1.0127*torch.exp(-5*Co*(P - Pref))

    def Kro(Sw, Somin, Swmin, Kro_):
        return Kro_*((1- Sw - Somin)/(1 - Somin - Swmin))**3

    # WATER PROPERTIES
    def rho_w(P, Cw, Pref, rho_w_std):
        return rho_w_std*torch.exp(Cw*(P - Pref))

    def mu_w(P, Cw, Pref):
        return 1.0*torch.exp(-0*Cw*(P - Pref))

    def Krw(Sw, Somin, Swmin, Krw_):
        return Krw_*((Sw - Swmin)/(1 - Somin - Swmin))**2

    def up(p, c):
        return (p[c[:, 0]] > p[c[:, 1]]).int()
    
    # MEDIA PROPERTIES:
    def dporositydp(P, Cr, Pref, phi):
        return Cr*phi*torch.exp(Cr*(P - Pref))

    # OIL PROPERTIES
    def drho_odp(P, Co, Pref, rho_o_std):
        return Co*rho_o_std*torch.exp(Co*(P - Pref))

    def dmu_odp(P, Co, Pref):
        return -5*Co*1.0127*torch.exp(-5*Co*(P - Pref))

    def dKrodSw(Sw, Somin, Swmin, Kro_):
        return -3*Kro_*((1- Sw - Somin)**2/(1 - Somin - Swmin)**3)

    # WATER PROPERTIES
    def drho_wdp(P, Cw, Pref, rho_w_std):
        return Cw*rho_w_std*torch.exp(Cw*(P - Pref))

    def dmu_wdp(P, Cw, Pref):
        return 0.0*torch.exp(-0*Cw*(P - Pref))

    def dKrwdSw(Sw, Somin, Swmin, Krw_):
        return 2*Krw_*((Sw - Swmin)/(1 - Somin - Swmin)**2)
    
class Transmissibility:
    
    def T_geometric(K, dx, dy, dz, cell_connections):
        return (dz*dy/dx) * 2 / (1/K[cell_connections[:, 0]] + 1/K[cell_connections[:, 1]])

    def T(mob, TGeo, cell_connections, a):
        return TGeo*(mob[cell_connections[:, 0]]*a + mob[cell_connections[:, 1]]*(1-a))
    
    def dTdp(dmobdp, TGeo, cell_connections, a):
        return TGeo*(dmobdp[cell_connections[:, 0]]*a + dmobdp[cell_connections[:, 1]]*(1-a))
    
    def dTds(dmobds, TGeo, cell_connections, a):
        return TGeo*(dmobds[cell_connections[:, 0]]*a + dmobds[cell_connections[:, 1]]*(1-a))
    
class Residual_Jacobian:
    def __init__(self,Dict, ProdWells, InjWells, cell_connections):

        self.ProdWells = ProdWells
        self.InjWells = InjWells
        self.cell_connections = cell_connections

        self.K, self.phi, self.rho_o_std, self.rho_w_std, self.Pref, self.Co, self.Cw, self.Cr, self.Swmin,  self.Somin, \
        self.BHP, self.rw, self.re, self.h, self.dx, self.dy, self.dz, self.dt, self.dV,\
        self.S, self.QinjW, self.Kro_,  self.Krw_= \
            Dict["K"],  Dict["phi"], Dict["rho_o_std"], Dict["rho_w_std"], Dict["Pref"], Dict["Co"], \
            Dict["Cw"], Dict["Cr"],  Dict["Swmin"],     Dict["Somin"],     Dict["BHP"],  Dict["rw"], \
            Dict["re"], Dict["h"],   Dict["dx"],        Dict["dy"],        Dict["dz"],   Dict["dt"], \
            Dict["dV"], Dict["S"],   Dict["QinjW"],     Dict["Kro_"],      Dict["Krw_"]
        
        self.properties = properties
        self.Transmissibility = Transmissibility
        
        self.TGeo = (self.Transmissibility.T_geometric(self.K, self.dx, self.dy, self.dz, cell_connections)).float()
        self.WI = ((2*torch.pi*self.K*self.h/(torch.log(self.re/self.rw) + self.S))*ProdWells).float()
        
    def Sink(self, mob, BHP, P):
        return self.WI*mob*(P - BHP)
    
    def dSdp(self, mob, dmobdp):
        return self.WI*(mob + dmobdp)

    def dSds(self, dmobds, BHP, P):
        return self.WI*dmobds*(P - BHP)

    def Accumilation(self, V, poro, rho, rho_std, Cf, Cr, P_hist, S_hist, dt, P, S, n):
        return (V*poro*rho/(rho_std*dt))*(S*(Cf + Cr)*(P - P_hist[n]) + (S - S_hist[n]))
    
    def dAdp(self, V, poro, rho, rho_std, drhodp, dporodp, Cf, Cr, P_hist, S_hist, dt, P, S, n):
        coef_A = V*poro*rho/(dt*rho_std)
        dcoef_Adp = (V/(dt*rho_std))*(drhodp*poro + dporodp*rho)
        return dcoef_Adp*(S*(Cf + Cr)*(P - P_hist[n]) + (S - S_hist[n])) + coef_A*S*(Cf + Cr)

    def dAds(self, V, poro, rho, rho_std, Cf, Cr, P_hist, dt, P, n):
        coef_A = V*poro*rho/(dt*rho_std)
        return coef_A*((Cf + Cr)*(P - P_hist[n]) + 1)

    def Flux(self, T, P, cell_connections, Flux_Holder):
        F_connections = T*(P[cell_connections[:, 1]] - P[cell_connections[:, 0]])
        Flux_Holder[cell_connections[:, 0], cell_connections[:, 1]] =  F_connections
        Flux_Holder[cell_connections[:, 1], cell_connections[:, 0]] = -F_connections
        return torch.sum(Flux_Holder, 1)
    
    def dFdp(self, T, dTdp, P, cell_connections, a):
        dp = P[cell_connections[:, 1]] - P[cell_connections[:, 0]] 
        
        return [-T + dTdp*dp*a, T + dTdp*dp*(1-a)]

    def dFds(self, T, dTds, P, cell_connections, a):
        dp = P[cell_connections[:, 1]] - P[cell_connections[:, 0]]
        return [ dTds*dp*a, dTds*dp*(1-a)]

    def calculate_Residual_Jacobian(self, X, C, P_hist, Sw_hist, n, N):
        Flux_Holder = torch.zeros((N, N)).to(device=device)
        Jac         = torch.zeros((N*2, N*2)).to(device=device)
        P = X[:N]
        Sw = X[N:]
    # Media Properties
        poro_    = self.properties.porosity(P, self.Cr, self.Pref, self.phi)
        dporodp_ = self.properties.dporositydp(P, self.Cr, self.Pref, self.phi)
    # Oil Properties
        rho_o_   = self.properties.rho_o(P, self.Co, self.Pref, self.rho_o_std)
        mu_o_    = self.properties.mu_o(P, self.Co, self.Pref)
        kro      = self.properties.Kro(Sw, self.Somin, self.Swmin, self.Kro_)
        mobo     = (kro*rho_o_/(mu_o_*self.rho_o_std))

        drhodp_o = self.properties.drho_odp(P, self.Co, self.Pref, self.rho_o_std)
        dmu_odp_ = self.properties.dmu_odp(P, self.Co, self.Pref)
        dKrodSw_ = self.properties.dKrodSw(Sw, self.Somin, self.Swmin, self.Kro_)
        dmobodp  = (kro/self.rho_o_std)*(rho_o_*(-1/mu_o_**2)*dmu_odp_  + (1/mu_o_)*drhodp_o)
        dmobodsw = (dKrodSw_*rho_o_/(mu_o_*self.rho_o_std))
    # Water Properties
        rho_w_   = self.properties.rho_w(P, self.Cw, self.Pref, self.rho_w_std)
        mu_w_    = self.properties.mu_w(P, self.Cw, self.Pref)
        krw      = self.properties.Krw(Sw, self.Somin, self.Swmin, self.Krw_)
        mobw     = (krw*rho_w_/(mu_w_*self.rho_w_std))

        drhodp_w = self.properties.drho_wdp(P, self.Cw, self.Pref, self.rho_w_std)
        dmu_wdp_ = self.properties.dmu_wdp(P, self.Cw, self.Pref)
        dKrwdSw_ = self.properties.dKrwdSw(Sw, self.Somin, self.Swmin, self.Krw_)
        dmobwdp  = (krw/self.rho_w_std)*(rho_w_*(-1/mu_w_**2)*dmu_wdp_  + (1/mu_w_)*drhodp_w)
        dmobwdsw = (dKrwdSw_*rho_w_/(mu_w_*self.rho_w_std))

    # T
        a = self.properties.up(P, self.cell_connections)
        T_o = C[0]*self.Transmissibility.T(mobo, self.TGeo, self.cell_connections, a)
        T_w = C[0]*self.Transmissibility.T(mobw, self.TGeo, self.cell_connections, a)

        #====================== RESIDUAL ======================#

    # SS 
        o_Q = C[0]*self.Sink(mobo, self.BHP, P)
        w_Q = C[0]*self.Sink(mobw, self.BHP, P) + (C[2]*self.QinjW*rho_w_/self.rho_w_std)*self.InjWells

    # A
        o_A = self.Accumilation(self.dV, poro_, rho_o_, self.rho_o_std, self.Co, self.Cr, P_hist, 1-Sw_hist, self.dt, P, 1-Sw, n)
        w_A = self.Accumilation(self.dV, poro_, rho_w_, self.rho_w_std, self.Cw, self.Cr, P_hist, Sw_hist, self.dt, P, Sw, n)
    
    # F
        o_F = self.Flux(T_o, P, self.cell_connections, Flux_Holder)
        w_F = self.Flux(T_w, P, self.cell_connections, Flux_Holder)
    # R
        o_R = o_F - o_A - o_Q
        w_R = w_F - w_A - w_Q

        #====================== JACOBIAN ======================#
    # DTDX
        dTdp_o = C[0]*self.Transmissibility.dTdp(dmobodp, self.TGeo, self.cell_connections, a)
        dTdp_w = C[0]*self.Transmissibility.dTdp(dmobwdp, self.TGeo, self.cell_connections, a)
        dTds_o = C[0]*self.Transmissibility.dTds(dmobodsw, self.TGeo, self.cell_connections, a)
        dTds_w = C[0]*self.Transmissibility.dTds(dmobwdsw, self.TGeo, self.cell_connections, a)
    # DFDX
        dFo_dp = self.dFdp(T_o, dTdp_o, P, self.cell_connections, a)
        dFw_dp = self.dFdp(T_w, dTdp_w, P, self.cell_connections, a)
        dFo_ds = self.dFds(T_o, dTds_o, P, self.cell_connections, a)
        dFw_ds = self.dFds(T_w, dTds_w, P, self.cell_connections, a)
    # DADX
        dAo_dp =  self.dAdp(self.dV, poro_, rho_o_, self.rho_o_std, drhodp_o, dporodp_, self.Co, self.Cr, P_hist, 1-Sw_hist, self.dt, P, 1-Sw, n)
        dAw_dp =  self.dAdp(self.dV, poro_, rho_w_, self.rho_w_std, drhodp_w, dporodp_, self.Cw, self.Cr, P_hist, Sw_hist,   self.dt, P,   Sw, n)
        dAo_ds = -self.dAds(self.dV, poro_, rho_o_, self.rho_o_std, self.Co, self.Cr, P_hist, self.dt, P, n)
        dAw_ds =  self.dAds(self.dV, poro_, rho_w_, self.rho_w_std, self.Cw, self.Cr, P_hist, self.dt, P, n)
    # DQDX
        dQo_dp = C[0]*self.dSdp(mobo, dmobodp)
        dQw_dp = C[0]*self.dSdp(mobw, dmobwdp) + (C[2]*self.QinjW*drhodp_w/self.rho_w_std)*self.InjWells
        dQo_ds = C[0]*self.dSds(dmobodsw, self.BHP, P)
        dQw_ds = C[0]*self.dSds(dmobwdsw, self.BHP, P)

    # J
        grid_0_all = self.cell_connections[:, 0]
        grid_1_all = self.cell_connections[:, 1]
        Jac[:N,:N][grid_0_all, grid_1_all]  =  dFo_dp[1]
        Jac[:N,:N][grid_1_all, grid_0_all]  = -dFo_dp[0]
        Flux_Holder[grid_0_all, grid_1_all] =  dFo_dp[0]
        Flux_Holder[grid_1_all, grid_0_all] = -dFo_dp[1]
        Jac[:N,:N][range(N), range(N)] = torch.sum(Flux_Holder, 1) - dAo_dp  - dQo_dp

        Jac[N:, :N][grid_0_all, grid_1_all] =  dFw_dp[1]
        Jac[N:, :N][grid_1_all, grid_0_all] = -dFw_dp[0]
        Flux_Holder[grid_0_all, grid_1_all] =  dFw_dp[0]
        Flux_Holder[grid_1_all, grid_0_all] = -dFw_dp[1]
        Jac[N:, :N][range(N), range(N)] = torch.sum(Flux_Holder, 1) - dAw_dp  - dQw_dp

        Jac[:N, N:][grid_0_all, grid_1_all] =  dFo_ds[1]
        Jac[:N, N:][grid_1_all, grid_0_all] = -dFo_ds[0]
        Flux_Holder[grid_0_all, grid_1_all] =  dFo_ds[0]
        Flux_Holder[grid_1_all, grid_0_all] = -dFo_ds[1]
        Jac[:N, N:][range(N), range(N)] = torch.sum(Flux_Holder, 1) - dAo_ds - dQo_ds

        Jac[N:, N:][grid_0_all, grid_1_all] =  dFw_ds[1]
        Jac[N:, N:][grid_1_all, grid_0_all] = -dFw_ds[0]
        Flux_Holder[grid_0_all, grid_1_all] =  dFw_ds[0]
        Flux_Holder[grid_1_all, grid_0_all] = -dFw_ds[1]
        Jac[N:, N:][range(N), range(N)] = torch.sum(Flux_Holder, 1) - dAw_ds - dQw_ds

        return Jac, torch.cat((o_R, w_R), dim = 0)