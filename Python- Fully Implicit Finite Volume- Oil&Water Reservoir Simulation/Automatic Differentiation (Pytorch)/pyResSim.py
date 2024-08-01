import torch

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
    

class Transmissibility:
    
    def T_geometric(K, dx, dy, dz, cell_connections):
        return (dz*dy/dx) * 2 / (1/K[cell_connections[:, 0]] + 1/K[cell_connections[:, 1]])

    def half_T(kr, rho,  mu, TGeo, cell_connections, a):
        return TGeo*((kr*rho/mu)[cell_connections[:, 0]]*a + (kr*rho/mu)[cell_connections[:, 1]]*(1-a))
    #def half_T(kr, rho,  mu, TGeo, cell_connections, a):
    #    mob = (kr*rho/mu)
    #    T_01 =  mob[cell_connections[:, 0]]
    #    T_01[a == 0] = mob[cell_connections[:, 1]][a == 0]
    #    return TGeo*T_01
class Residual:
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
        
    def Sink(self, kr, rho, mu, BHP, P):
        return self.WI * (kr*rho/mu)*(P - BHP)

    def Accumilation(self, V, poro, rho, Ca, Cr, P_hist, S_hist, dt, P, S, n):
        return (V*poro*rho/dt)*(S*(Ca + Cr)*(P - P_hist[n]) + (S - S_hist[n]))

    def Flux(self, half_T, P, cell_connections, Flux_Holder):
        F_connections = half_T*(P[cell_connections[:, 1]] - P[cell_connections[:, 0]])
        Flux_Holder[cell_connections[:, 0], cell_connections[:, 1]] =  F_connections
        Flux_Holder[cell_connections[:, 1], cell_connections[:, 0]] = -F_connections
        return torch.sum(Flux_Holder, 1)

    def calculate_Residual(self, X, C, P_hist, Sw_hist, n, Flux_Holder):
        P = X[:225]
        Sw = X[225:]
    # Media Properties
        poro_ = self.properties.porosity(P, self.Cr, self.Pref, self.phi)
    # Oil Properties
        rho_o_   = self.properties.rho_o(P, self.Co, self.Pref, self.rho_o_std)
        mu_o_ = self.properties.mu_o(P, self.Co, self.Pref)
        kro   = self.properties.Kro(Sw, self.Somin, self.Swmin, self.Kro_)

    # Water Properties
        rho_w_   = self.properties.rho_w(P, self.Cw, self.Pref, self.rho_w_std)
        mu_w_ = self.properties.mu_w(P, self.Cw, self.Pref)
        krw   = self.properties.Krw(Sw, self.Somin, self.Swmin, self.Krw_)
    # Half Tranmissibility
        a = self.properties.up(P, self.cell_connections)
        half_T_o = self.Transmissibility.half_T(kro, rho_o_,  mu_o_, self.TGeo, self.cell_connections, a)
        half_T_w = self.Transmissibility.half_T(krw, rho_w_, mu_w_, self.TGeo, self.cell_connections, a)

    # SS 
        o_Q = self.Sink(kro, rho_o_, mu_o_, self.BHP, P)*C[0]
        w_Q = self.Sink(krw, rho_w_, mu_w_, self.BHP, P)*C[0] + (C[2]*self.QinjW*rho_w_)*self.InjWells

    # A
        o_A = self.Accumilation(self.dV, poro_, rho_o_, self.Co, self.Cr, P_hist, 1-Sw_hist, self.dt, P, 1-Sw, n)
        w_A = self.Accumilation(self.dV, poro_, rho_w_, self.Cw, self.Cr, P_hist, Sw_hist, self.dt, P, Sw, n)
    
    # F
        o_F = C[0]*self.Flux(half_T_o, P, self.cell_connections, Flux_Holder)
        w_F = C[0]*self.Flux(half_T_w, P, self.cell_connections, Flux_Holder)


        return torch.cat((o_F - o_A - o_Q, w_F - w_A - w_Q), dim = 0)