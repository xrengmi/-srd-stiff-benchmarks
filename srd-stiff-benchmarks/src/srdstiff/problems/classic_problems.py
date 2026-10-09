import numpy as np

PROBLEMS = {}


class Sphere10Objective:
    name = "SPHERE10"
    n = 10
    
    @property
    def x_init(self):
        return np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
        return x0**2 + x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2 + x7**2 + x8**2 + x9**2
        
    def grad(self, x):
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
        g = np.zeros(10, dtype=np.float64)
        g[0] = 2*x0
        g[1] = 2*x1
        g[2] = 2*x2
        g[3] = 2*x3
        g[4] = 2*x4
        g[5] = 2*x5
        g[6] = 2*x6
        g[7] = 2*x7
        g[8] = 2*x8
        g[9] = 2*x9
        return g
        
    def hess_vec(self, x, v):
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9 = v
        hv = np.zeros(10, dtype=np.float64)
        hv[0] = 2*v0
        hv[1] = 2*v1
        hv[2] = 2*v2
        hv[3] = 2*v3
        hv[4] = 2*v4
        hv[5] = 2*v5
        hv[6] = 2*v6
        hv[7] = 2*v7
        hv[8] = 2*v8
        hv[9] = 2*v9
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["SPHERE10"] = Sphere10Objective


class BealeObjective:
    name = "BEALE"
    n = 2
    
    @property
    def x_init(self):
        return np.array([1.0, 1.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1 = x
        return (x0*x1 - x0 + 1.5)**2 + (x0*x1**2 - x0 + 2.25)**2 + (x0*x1**3 - x0 + 2.625)**2
        
    def grad(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=np.float64)
        g[0] = (2*x1 - 2)*(x0*x1 - x0 + 1.5) + (2*x1**2 - 2)*(x0*x1**2 - x0 + 2.25) + (2*x1**3 - 2)*(x0*x1**3 - x0 + 2.625)
        g[1] = 6*x0*x1**2*(x0*x1**3 - x0 + 2.625) + 4*x0*x1*(x0*x1**2 - x0 + 2.25) + 2*x0*(x0*x1 - x0 + 1.5)
        return g
        
    def hess_vec(self, x, v):
        x0, x1 = x
        v0, v1 = v
        hv = np.zeros(2, dtype=np.float64)
        hv[0] = v0*((x1 - 1)*(2*x1 - 2) + (x1**2 - 1)*(2*x1**2 - 2) + (x1**3 - 1)*(2*x1**3 - 2)) + v1*(3*x0*x1**2*(2*x1**3 - 2) + 2*x0*x1*(2*x1**2 - 2) + 2*x0*x1 + x0*(2*x1 - 2) - 2*x0 + 6*x1**2*(x0*x1**3 - x0 + 2.625) + 4*x1*(x0*x1**2 - x0 + 2.25) + 3.0)
        hv[1] = v0*(6*x0*x1**2*(x1**3 - 1) + 4*x0*x1*(x1**2 - 1) + 2*x0*x1 + 2*x0*(x1 - 1) - 2*x0 + 6*x1**2*(x0*x1**3 - x0 + 2.625) + 4*x1*(x0*x1**2 - x0 + 2.25) + 3.0) + v1*(18*x0**2*x1**4 + 8*x0**2*x1**2 + 2*x0**2 + 12*x0*x1*(x0*x1**3 - x0 + 2.625) + 4*x0*(x0*x1**2 - x0 + 2.25))
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["BEALE"] = BealeObjective


class BoothObjective:
    name = "BOOTH"
    n = 2
    
    @property
    def x_init(self):
        return np.array([1.0, 1.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1 = x
        return (x0 + 2*x1 - 7)**2 + (2*x0 + x1 - 5)**2
        
    def grad(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=np.float64)
        g[0] = 10*x0 + 8*x1 - 34
        g[1] = 8*x0 + 10*x1 - 38
        return g
        
    def hess_vec(self, x, v):
        x0, x1 = x
        v0, v1 = v
        hv = np.zeros(2, dtype=np.float64)
        hv[0] = 10*v0 + 8*v1
        hv[1] = 8*v0 + 10*v1
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["BOOTH"] = BoothObjective


class MatyasObjective:
    name = "MATYAS"
    n = 2
    
    @property
    def x_init(self):
        return np.array([1.0, 1.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1 = x
        return 0.26*x0**2 - 0.48*x0*x1 + 0.26*x1**2
        
    def grad(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=np.float64)
        g[0] = 0.52*x0 - 0.48*x1
        g[1] = -0.48*x0 + 0.52*x1
        return g
        
    def hess_vec(self, x, v):
        x0, x1 = x
        v0, v1 = v
        hv = np.zeros(2, dtype=np.float64)
        hv[0] = 0.52*v0 - 0.48*v1
        hv[1] = -0.48*v0 + 0.52*v1
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["MATYAS"] = MatyasObjective


class HimmelblauObjective:
    name = "HIMMELBLAU"
    n = 2
    
    @property
    def x_init(self):
        return np.array([1.0, 1.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1 = x
        return (x0 + x1**2 - 7)**2 + (x0**2 + x1 - 11)**2
        
    def grad(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=np.float64)
        g[0] = 4*x0*(x0**2 + x1 - 11) + 2*x0 + 2*x1**2 - 14
        g[1] = 2*x0**2 + 4*x1*(x0 + x1**2 - 7) + 2*x1 - 22
        return g
        
    def hess_vec(self, x, v):
        x0, x1 = x
        v0, v1 = v
        hv = np.zeros(2, dtype=np.float64)
        hv[0] = v0*(12*x0**2 + 4*x1 - 42) + v1*(4*x0 + 4*x1)
        hv[1] = v0*(4*x0 + 4*x1) + v1*(4*x0 + 12*x1**2 - 26)
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["HIMMELBLAU"] = HimmelblauObjective


class Ackley2Objective:
    name = "ACKLEY2"
    n = 2
    
    @property
    def x_init(self):
        return np.array([1.0, 1.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1 = x
        return -np.exp(0.5*np.cos(2*np.pi*x0) + 0.5*np.cos(2*np.pi*x1)) + np.e + 20 - 20*np.exp(-0.2*np.sqrt(0.5*x0**2 + 0.5*x1**2))
        
    def grad(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=np.float64)
        g[0] = 2.0*x0*np.exp(-0.2*np.sqrt(0.5*x0**2 + 0.5*x1**2))/np.sqrt(0.5*x0**2 + 0.5*x1**2) + 1.0*np.pi*np.exp(0.5*np.cos(2*np.pi*x0) + 0.5*np.cos(2*np.pi*x1))*np.sin(2*np.pi*x0)
        g[1] = 2.0*x1*np.exp(-0.2*np.sqrt(0.5*x0**2 + 0.5*x1**2))/np.sqrt(0.5*x0**2 + 0.5*x1**2) + 1.0*np.pi*np.exp(0.5*np.cos(2*np.pi*x0) + 0.5*np.cos(2*np.pi*x1))*np.sin(2*np.pi*x1)
        return g
        
    def hess_vec(self, x, v):
        x0, x1 = x
        v0, v1 = v
        hv = np.zeros(2, dtype=np.float64)
        hv[0] = v0*(-0.2*x0**2*np.exp(-0.2*np.sqrt(0.5*x0**2 + 0.5*x1**2))/(0.5*x0**2 + 0.5*x1**2) - 1.0*x0**2*np.exp(-0.2*np.sqrt(0.5*x0**2 + 0.5*x1**2))/(0.5*x0**2 + 0.5*x1**2)**(3/2) - 1.0*np.pi**2*np.exp(0.5*np.cos(2*np.pi*x0) + 0.5*np.cos(2*np.pi*x1))*np.sin(2*np.pi*x0)**2 + 2.0*np.pi**2*np.exp(0.5*np.cos(2*np.pi*x0) + 0.5*np.cos(2*np.pi*x1))*np.cos(2*np.pi*x0) + 2.0*np.exp(-0.2*np.sqrt(0.5*x0**2 + 0.5*x1**2))/np.sqrt(0.5*x0**2 + 0.5*x1**2)) + v1*(-0.2*x0*x1*np.exp(-0.2*np.sqrt(0.5*x0**2 + 0.5*x1**2))/(0.5*x0**2 + 0.5*x1**2) - 1.0*x0*x1*np.exp(-0.2*np.sqrt(0.5*x0**2 + 0.5*x1**2))/(0.5*x0**2 + 0.5*x1**2)**(3/2) - 1.0*np.pi**2*np.exp(0.5*np.cos(2*np.pi*x0) + 0.5*np.cos(2*np.pi*x1))*np.sin(2*np.pi*x0)*np.sin(2*np.pi*x1))
        hv[1] = v0*(-0.2*x0*x1*np.exp(-0.2*np.sqrt(0.5*x0**2 + 0.5*x1**2))/(0.5*x0**2 + 0.5*x1**2) - 1.0*x0*x1*np.exp(-0.2*np.sqrt(0.5*x0**2 + 0.5*x1**2))/(0.5*x0**2 + 0.5*x1**2)**(3/2) - 1.0*np.pi**2*np.exp(0.5*np.cos(2*np.pi*x0) + 0.5*np.cos(2*np.pi*x1))*np.sin(2*np.pi*x0)*np.sin(2*np.pi*x1)) + v1*(-0.2*x1**2*np.exp(-0.2*np.sqrt(0.5*x0**2 + 0.5*x1**2))/(0.5*x0**2 + 0.5*x1**2) - 1.0*x1**2*np.exp(-0.2*np.sqrt(0.5*x0**2 + 0.5*x1**2))/(0.5*x0**2 + 0.5*x1**2)**(3/2) - 1.0*np.pi**2*np.exp(0.5*np.cos(2*np.pi*x0) + 0.5*np.cos(2*np.pi*x1))*np.sin(2*np.pi*x1)**2 + 2.0*np.pi**2*np.exp(0.5*np.cos(2*np.pi*x0) + 0.5*np.cos(2*np.pi*x1))*np.cos(2*np.pi*x1) + 2.0*np.exp(-0.2*np.sqrt(0.5*x0**2 + 0.5*x1**2))/np.sqrt(0.5*x0**2 + 0.5*x1**2))
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["ACKLEY2"] = Ackley2Objective


class Rosenbr10Objective:
    name = "ROSENBR10"
    n = 10
    
    @property
    def x_init(self):
        return np.array([-1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
        return (1 - x0)**2 + (1 - x1)**2 + (1 - x2)**2 + (1 - x3)**2 + (1 - x4)**2 + (1 - x5)**2 + (1 - x6)**2 + (1 - x7)**2 + (1 - x8)**2 + 100*(-x0**2 + x1)**2 + 100*(-x1**2 + x2)**2 + 100*(-x2**2 + x3)**2 + 100*(-x3**2 + x4)**2 + 100*(-x4**2 + x5)**2 + 100*(-x5**2 + x6)**2 + 100*(-x6**2 + x7)**2 + 100*(-x7**2 + x8)**2 + 100*(-x8**2 + x9)**2
        
    def grad(self, x):
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
        g = np.zeros(10, dtype=np.float64)
        g[0] = -400*x0*(-x0**2 + x1) + 2*x0 - 2
        g[1] = -200*x0**2 - 400*x1*(-x1**2 + x2) + 202*x1 - 2
        g[2] = -200*x1**2 - 400*x2*(-x2**2 + x3) + 202*x2 - 2
        g[3] = -200*x2**2 - 400*x3*(-x3**2 + x4) + 202*x3 - 2
        g[4] = -200*x3**2 - 400*x4*(-x4**2 + x5) + 202*x4 - 2
        g[5] = -200*x4**2 - 400*x5*(-x5**2 + x6) + 202*x5 - 2
        g[6] = -200*x5**2 - 400*x6*(-x6**2 + x7) + 202*x6 - 2
        g[7] = -200*x6**2 - 400*x7*(-x7**2 + x8) + 202*x7 - 2
        g[8] = -200*x7**2 - 400*x8*(-x8**2 + x9) + 202*x8 - 2
        g[9] = -200*x8**2 + 200*x9
        return g
        
    def hess_vec(self, x, v):
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9 = v
        hv = np.zeros(10, dtype=np.float64)
        hv[0] = v0*(1200*x0**2 - 400*x1 + 2) - 400*v1*x0
        hv[1] = -400*v0*x0 + v1*(1200*x1**2 - 400*x2 + 202) - 400*v2*x1
        hv[2] = -400*v1*x1 + v2*(1200*x2**2 - 400*x3 + 202) - 400*v3*x2
        hv[3] = -400*v2*x2 + v3*(1200*x3**2 - 400*x4 + 202) - 400*v4*x3
        hv[4] = -400*v3*x3 + v4*(1200*x4**2 - 400*x5 + 202) - 400*v5*x4
        hv[5] = -400*v4*x4 + v5*(1200*x5**2 - 400*x6 + 202) - 400*v6*x5
        hv[6] = -400*v5*x5 + v6*(1200*x6**2 - 400*x7 + 202) - 400*v7*x6
        hv[7] = -400*v6*x6 + v7*(1200*x7**2 - 400*x8 + 202) - 400*v8*x7
        hv[8] = -400*v7*x7 + v8*(1200*x8**2 - 400*x9 + 202) - 400*v9*x8
        hv[9] = -400*v8*x8 + 200*v9
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["ROSENBR10"] = Rosenbr10Objective


class Dixonprice5Objective:
    name = "DIXONPRICE5"
    n = 5
    
    @property
    def x_init(self):
        return np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1, x2, x3, x4 = x
        return 2*(-x0 + 2*x1**2)**2 + (x0 - 1)**2 + 3*(-x1 + 2*x2**2)**2 + 4*(-x2 + 2*x3**2)**2 + 5*(-x3 + 2*x4**2)**2
        
    def grad(self, x):
        x0, x1, x2, x3, x4 = x
        g = np.zeros(5, dtype=np.float64)
        g[0] = 6*x0 - 8*x1**2 - 2
        g[1] = 16*x1*(-x0 + 2*x1**2) + 6*x1 - 12*x2**2
        g[2] = 24*x2*(-x1 + 2*x2**2) + 8*x2 - 16*x3**2
        g[3] = 32*x3*(-x2 + 2*x3**2) + 10*x3 - 20*x4**2
        g[4] = 40*x4*(-x3 + 2*x4**2)
        return g
        
    def hess_vec(self, x, v):
        x0, x1, x2, x3, x4 = x
        v0, v1, v2, v3, v4 = v
        hv = np.zeros(5, dtype=np.float64)
        hv[0] = 6*v0 - 16*v1*x1
        hv[1] = -16*v0*x1 + v1*(-16*x0 + 96*x1**2 + 6) - 24*v2*x2
        hv[2] = -24*v1*x2 + v2*(-24*x1 + 144*x2**2 + 8) - 32*v3*x3
        hv[3] = -32*v2*x3 + v3*(-32*x2 + 192*x3**2 + 10) - 40*v4*x4
        hv[4] = -40*v3*x4 + v4*(-40*x3 + 240*x4**2)
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["DIXONPRICE5"] = Dixonprice5Objective


class Zakharov5Objective:
    name = "ZAKHAROV5"
    n = 5
    
    @property
    def x_init(self):
        return np.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1, x2, x3, x4 = x
        return x0**2 + x1**2 + x2**2 + x3**2 + x4**2 + (0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**4 + (0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2
        
    def grad(self, x):
        x0, x1, x2, x3, x4 = x
        g = np.zeros(5, dtype=np.float64)
        g[0] = 2.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4 + 2.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**3
        g[1] = 1.0*x0 + 4.0*x1 + 3.0*x2 + 4.0*x3 + 5.0*x4 + 4.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**3
        g[2] = 1.5*x0 + 3.0*x1 + 6.5*x2 + 6.0*x3 + 7.5*x4 + 6.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**3
        g[3] = 2.0*x0 + 4.0*x1 + 6.0*x2 + 10.0*x3 + 10.0*x4 + 8.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**3
        g[4] = 2.5*x0 + 5.0*x1 + 7.5*x2 + 10.0*x3 + 14.5*x4 + 10.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**3
        return g
        
    def hess_vec(self, x, v):
        x0, x1, x2, x3, x4 = x
        v0, v1, v2, v3, v4 = v
        hv = np.zeros(5, dtype=np.float64)
        hv[0] = v0*(3.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 2.5) + v1*(6.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 1.0) + v2*(9.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 1.5) + v3*(12.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 2.0) + v4*(15.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 2.5)
        hv[1] = v0*(6.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 1.0) + v1*(12.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 4.0) + v2*(18.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 3.0) + v3*(24.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 4.0) + v4*(30.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 5.0)
        hv[2] = v0*(9.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 1.5) + v1*(18.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 3.0) + v2*(27.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 6.5) + v3*(36.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 6.0) + v4*(45.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 7.5)
        hv[3] = v0*(12.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 2.0) + v1*(24.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 4.0) + v2*(36.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 6.0) + v3*(48.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 10.0) + v4*(60.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 10.0)
        hv[4] = v0*(15.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 2.5) + v1*(30.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 5.0) + v2*(45.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 7.5) + v3*(60.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 10.0) + v4*(75.0*(0.5*x0 + 1.0*x1 + 1.5*x2 + 2.0*x3 + 2.5*x4)**2 + 14.5)
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["ZAKHAROV5"] = Zakharov5Objective


class Trid6Objective:
    name = "TRID6"
    n = 6
    
    @property
    def x_init(self):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1, x2, x3, x4, x5 = x
        return -x0*x1 - x1*x2 - x2*x3 - x3*x4 - x4*x5 + (x0 - 1)**2 + (x1 - 1)**2 + (x2 - 1)**2 + (x3 - 1)**2 + (x4 - 1)**2 + (x5 - 1)**2
        
    def grad(self, x):
        x0, x1, x2, x3, x4, x5 = x
        g = np.zeros(6, dtype=np.float64)
        g[0] = 2*x0 - x1 - 2
        g[1] = -x0 + 2*x1 - x2 - 2
        g[2] = -x1 + 2*x2 - x3 - 2
        g[3] = -x2 + 2*x3 - x4 - 2
        g[4] = -x3 + 2*x4 - x5 - 2
        g[5] = -x4 + 2*x5 - 2
        return g
        
    def hess_vec(self, x, v):
        x0, x1, x2, x3, x4, x5 = x
        v0, v1, v2, v3, v4, v5 = v
        hv = np.zeros(6, dtype=np.float64)
        hv[0] = 2*v0 - v1
        hv[1] = -v0 + 2*v1 - v2
        hv[2] = -v1 + 2*v2 - v3
        hv[3] = -v2 + 2*v3 - v4
        hv[4] = -v3 + 2*v4 - v5
        hv[5] = -v4 + 2*v5
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["TRID6"] = Trid6Objective


class Bohach1Objective:
    name = "BOHACH1"
    n = 2
    
    @property
    def x_init(self):
        return np.array([1.0, 1.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1 = x
        return x0**2 + 2*x1**2 - 0.3*np.cos(3*np.pi*x0) - 0.4*np.cos(4*np.pi*x1) + 0.7
        
    def grad(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=np.float64)
        g[0] = 2*x0 + 0.9*np.pi*np.sin(3*np.pi*x0)
        g[1] = 4*x1 + 1.6*np.pi*np.sin(4*np.pi*x1)
        return g
        
    def hess_vec(self, x, v):
        x0, x1 = x
        v0, v1 = v
        hv = np.zeros(2, dtype=np.float64)
        hv[0] = v0*(2.7*np.pi**2*np.cos(3*np.pi*x0) + 2)
        hv[1] = v1*(6.4*np.pi**2*np.cos(4*np.pi*x1) + 4)
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["BOHACH1"] = Bohach1Objective


class Bohach2Objective:
    name = "BOHACH2"
    n = 2
    
    @property
    def x_init(self):
        return np.array([1.0, 1.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1 = x
        return x0**2 + 2*x1**2 - 0.3*np.cos(3*np.pi*x0)*np.cos(4*np.pi*x1) + 0.3
        
    def grad(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=np.float64)
        g[0] = 2*x0 + 0.9*np.pi*np.sin(3*np.pi*x0)*np.cos(4*np.pi*x1)
        g[1] = 4*x1 + 1.2*np.pi*np.sin(4*np.pi*x1)*np.cos(3*np.pi*x0)
        return g
        
    def hess_vec(self, x, v):
        x0, x1 = x
        v0, v1 = v
        hv = np.zeros(2, dtype=np.float64)
        hv[0] = v0*(2.7*np.pi**2*np.cos(3*np.pi*x0)*np.cos(4*np.pi*x1) + 2) - 3.6*np.pi**2*v1*np.sin(3*np.pi*x0)*np.sin(4*np.pi*x1)
        hv[1] = -3.6*np.pi**2*v0*np.sin(3*np.pi*x0)*np.sin(4*np.pi*x1) + v1*(4.8*np.pi**2*np.cos(3*np.pi*x0)*np.cos(4*np.pi*x1) + 4)
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["BOHACH2"] = Bohach2Objective


class Styblinski4Objective:
    name = "STYBLINSKI4"
    n = 4
    
    @property
    def x_init(self):
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1, x2, x3 = x
        return 0.5*x0**4 - 8.0*x0**2 + 2.5*x0 + 0.5*x1**4 - 8.0*x1**2 + 2.5*x1 + 0.5*x2**4 - 8.0*x2**2 + 2.5*x2 + 0.5*x3**4 - 8.0*x3**2 + 2.5*x3
        
    def grad(self, x):
        x0, x1, x2, x3 = x
        g = np.zeros(4, dtype=np.float64)
        g[0] = 2.0*x0**3 - 16.0*x0 + 2.5
        g[1] = 2.0*x1**3 - 16.0*x1 + 2.5
        g[2] = 2.0*x2**3 - 16.0*x2 + 2.5
        g[3] = 2.0*x3**3 - 16.0*x3 + 2.5
        return g
        
    def hess_vec(self, x, v):
        x0, x1, x2, x3 = x
        v0, v1, v2, v3 = v
        hv = np.zeros(4, dtype=np.float64)
        hv[0] = v0*(6.0*x0**2 - 16.0)
        hv[1] = v1*(6.0*x1**2 - 16.0)
        hv[2] = v2*(6.0*x2**2 - 16.0)
        hv[3] = v3*(6.0*x3**2 - 16.0)
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["STYBLINSKI4"] = Styblinski4Objective


class Sumsquares10Objective:
    name = "SUMSQUARES10"
    n = 10
    
    @property
    def x_init(self):
        return np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
        return x0**2 + 2*x1**2 + 3*x2**2 + 4*x3**2 + 5*x4**2 + 6*x5**2 + 7*x6**2 + 8*x7**2 + 9*x8**2 + 10*x9**2
        
    def grad(self, x):
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
        g = np.zeros(10, dtype=np.float64)
        g[0] = 2*x0
        g[1] = 4*x1
        g[2] = 6*x2
        g[3] = 8*x3
        g[4] = 10*x4
        g[5] = 12*x5
        g[6] = 14*x6
        g[7] = 16*x7
        g[8] = 18*x8
        g[9] = 20*x9
        return g
        
    def hess_vec(self, x, v):
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = x
        v0, v1, v2, v3, v4, v5, v6, v7, v8, v9 = v
        hv = np.zeros(10, dtype=np.float64)
        hv[0] = 2*v0
        hv[1] = 4*v1
        hv[2] = 6*v2
        hv[3] = 8*v3
        hv[4] = 10*v4
        hv[5] = 12*v5
        hv[6] = 14*v6
        hv[7] = 16*v7
        hv[8] = 18*v8
        hv[9] = 20*v9
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["SUMSQUARES10"] = Sumsquares10Objective


class MccormickObjective:
    name = "MCCORMICK"
    n = 2
    
    @property
    def x_init(self):
        return np.array([-0.5, 0.5], dtype=np.float64)
        
    def f(self, x):
        x0, x1 = x
        return -1.5*x0 + 2.5*x1 + (x0 - x1)**2 + np.sin(x0 + x1) + 1.0
        
    def grad(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=np.float64)
        g[0] = 2*x0 - 2*x1 + np.cos(x0 + x1) - 1.5
        g[1] = -2*x0 + 2*x1 + np.cos(x0 + x1) + 2.5
        return g
        
    def hess_vec(self, x, v):
        x0, x1 = x
        v0, v1 = v
        hv = np.zeros(2, dtype=np.float64)
        hv[0] = v0*(2 - np.sin(x0 + x1)) + v1*(-np.sin(x0 + x1) - 2)
        hv[1] = v0*(-np.sin(x0 + x1) - 2) + v1*(2 - np.sin(x0 + x1))
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["MCCORMICK"] = MccormickObjective


class Dixmaana6Objective:
    name = "DIXMAANA6"
    n = 6
    
    @property
    def x_init(self):
        return np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1, x2, x3, x4, x5 = x
        return x0**2*(x1 + x2)**2 + x0**2 + x1**2*(x2 + x3)**2 + x1**2 + x2**2*(x3 + x4)**2 + x2**2 + x3**2*(x4 + x5)**2 + x3**2 + x4**2 + x5**2 + 1.0
        
    def grad(self, x):
        x0, x1, x2, x3, x4, x5 = x
        g = np.zeros(6, dtype=np.float64)
        g[0] = 2*x0*(x1 + x2)**2 + 2*x0
        g[1] = x0**2*(2*x1 + 2*x2) + 2*x1*(x2 + x3)**2 + 2*x1
        g[2] = x0**2*(2*x1 + 2*x2) + x1**2*(2*x2 + 2*x3) + 2*x2*(x3 + x4)**2 + 2*x2
        g[3] = x1**2*(2*x2 + 2*x3) + x2**2*(2*x3 + 2*x4) + 2*x3*(x4 + x5)**2 + 2*x3
        g[4] = x2**2*(2*x3 + 2*x4) + x3**2*(2*x4 + 2*x5) + 2*x4
        g[5] = x3**2*(2*x4 + 2*x5) + 2*x5
        return g
        
    def hess_vec(self, x, v):
        x0, x1, x2, x3, x4, x5 = x
        v0, v1, v2, v3, v4, v5 = v
        hv = np.zeros(6, dtype=np.float64)
        hv[0] = v0*(2*(x1 + x2)**2 + 2) + 2*v1*x0*(2*x1 + 2*x2) + 2*v2*x0*(2*x1 + 2*x2)
        hv[1] = 2*v0*x0*(2*x1 + 2*x2) + v1*(2*x0**2 + 2*(x2 + x3)**2 + 2) + v2*(2*x0**2 + 2*x1*(2*x2 + 2*x3)) + 2*v3*x1*(2*x2 + 2*x3)
        hv[2] = 2*v0*x0*(2*x1 + 2*x2) + v1*(2*x0**2 + 2*x1*(2*x2 + 2*x3)) + v2*(2*x0**2 + 2*x1**2 + 2*(x3 + x4)**2 + 2) + v3*(2*x1**2 + 2*x2*(2*x3 + 2*x4)) + 2*v4*x2*(2*x3 + 2*x4)
        hv[3] = 2*v1*x1*(2*x2 + 2*x3) + v2*(2*x1**2 + 2*x2*(2*x3 + 2*x4)) + v3*(2*x1**2 + 2*x2**2 + 2*(x4 + x5)**2 + 2) + v4*(2*x2**2 + 2*x3*(2*x4 + 2*x5)) + 2*v5*x3*(2*x4 + 2*x5)
        hv[4] = 2*v2*x2*(2*x3 + 2*x4) + v3*(2*x2**2 + 2*x3*(2*x4 + 2*x5)) + v4*(2*x2**2 + 2*x3**2 + 2) + 2*v5*x3**2
        hv[5] = 2*v3*x3*(2*x4 + 2*x5) + 2*v4*x3**2 + v5*(2*x3**2 + 2)
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["DIXMAANA6"] = Dixmaana6Objective


class PowellObjective:
    name = "POWELL"
    n = 4
    
    @property
    def x_init(self):
        return np.array([3.0, -1.0, 0.0, 1.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1, x2, x3 = x
        return (x0 + 10*x1)**2 + 10*(x0 - x3)**4 + (x1 - 2*x2)**4 + 5*(x2 - x3)**2
        
    def grad(self, x):
        x0, x1, x2, x3 = x
        g = np.zeros(4, dtype=np.float64)
        g[0] = 2*x0 + 20*x1 + 40*(x0 - x3)**3
        g[1] = 20*x0 + 200*x1 + 4*(x1 - 2*x2)**3
        g[2] = 10*x2 - 10*x3 - 8*(x1 - 2*x2)**3
        g[3] = -10*x2 + 10*x3 - 40*(x0 - x3)**3
        return g
        
    def hess_vec(self, x, v):
        x0, x1, x2, x3 = x
        v0, v1, v2, v3 = v
        hv = np.zeros(4, dtype=np.float64)
        hv[0] = v0*(120*(x0 - x3)**2 + 2) + 20*v1 - 120*v3*(x0 - x3)**2
        hv[1] = 20*v0 + v1*(12*(x1 - 2*x2)**2 + 200) - 24*v2*(x1 - 2*x2)**2
        hv[2] = -24*v1*(x1 - 2*x2)**2 + v2*(48*(x1 - 2*x2)**2 + 10) - 10*v3
        hv[3] = -120*v0*(x0 - x3)**2 - 10*v2 + v3*(120*(x0 - x3)**2 + 10)
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["POWELL"] = PowellObjective


class LeonObjective:
    name = "LEON"
    n = 2
    
    @property
    def x_init(self):
        return np.array([-1.2, 1.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1 = x
        return (1 - x0)**2 + 100*(-x0**3 + x1)**2
        
    def grad(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=np.float64)
        g[0] = -600*x0**2*(-x0**3 + x1) + 2*x0 - 2
        g[1] = -200*x0**3 + 200*x1
        return g
        
    def hess_vec(self, x, v):
        x0, x1 = x
        v0, v1 = v
        hv = np.zeros(2, dtype=np.float64)
        hv[0] = v0*(1800*x0**4 - 1200*x0*(-x0**3 + x1) + 2) - 600*v1*x0**2
        hv[1] = -600*v0*x0**2 + 200*v1
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["LEON"] = LeonObjective


class Camel3Objective:
    name = "CAMEL3"
    n = 2
    
    @property
    def x_init(self):
        return np.array([2.0, 2.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1 = x
        return x0**6/6 - 1.05*x0**4 + 2*x0**2 + x0*x1 + x1**2
        
    def grad(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=np.float64)
        g[0] = x0**5 - 4.2*x0**3 + 4*x0 + x1
        g[1] = x0 + 2*x1
        return g
        
    def hess_vec(self, x, v):
        x0, x1 = x
        v0, v1 = v
        hv = np.zeros(2, dtype=np.float64)
        hv[0] = v0*(5*x0**4 - 12.6*x0**2 + 4) + v1
        hv[1] = v0 + 2*v1
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["CAMEL3"] = Camel3Objective


class Camel6Objective:
    name = "CAMEL6"
    n = 2
    
    @property
    def x_init(self):
        return np.array([2.0, -2.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1 = x
        return x0**2*(x0**4/3 - 2.1*x0**2 + 4) + x0*x1 + x1**2*(4*x1**2 - 4)
        
    def grad(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=np.float64)
        g[0] = x0**2*(4*x0**3/3 - 4.2*x0) + 2*x0*(x0**4/3 - 2.1*x0**2 + 4) + x1
        g[1] = x0 + 8*x1**3 + 2*x1*(4*x1**2 - 4)
        return g
        
    def hess_vec(self, x, v):
        x0, x1 = x
        v0, v1 = v
        hv = np.zeros(2, dtype=np.float64)
        hv[0] = v0*(2*x0**4/3 + x0**2*(4*x0**2 - 4.2) - 4.2*x0**2 + 4*x0*(4*x0**3/3 - 4.2*x0) + 8) + v1
        hv[1] = v0 + v1*(48*x1**2 - 8)
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["CAMEL6"] = Camel6Objective


class ChunhuiObjective:
    name = "CHUNHUI"
    n = 2
    
    @property
    def x_init(self):
        return np.array([1.0, 1.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1 = x
        return x0**2 + x0*x1 - 3*x0 + x1**2
        
    def grad(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=np.float64)
        g[0] = 2*x0 + x1 - 3
        g[1] = x0 + 2*x1
        return g
        
    def hess_vec(self, x, v):
        x0, x1 = x
        v0, v1 = v
        hv = np.zeros(2, dtype=np.float64)
        hv[0] = 2*v0 + v1
        hv[1] = v0 + 2*v1
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["CHUNHUI"] = ChunhuiObjective


class DropwaveObjective:
    name = "DROPWAVE"
    n = 2
    
    @property
    def x_init(self):
        return np.array([5.0, 5.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1 = x
        return (-np.cos(12*np.sqrt(x0**2 + x1**2)) - 1)/(0.5*x0**2 + 0.5*x1**2 + 2)
        
    def grad(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=np.float64)
        g[0] = -1.0*x0*(-np.cos(12*np.sqrt(x0**2 + x1**2)) - 1)/(0.5*x0**2 + 0.5*x1**2 + 2)**2 + 12*x0*np.sin(12*np.sqrt(x0**2 + x1**2))/(np.sqrt(x0**2 + x1**2)*(0.5*x0**2 + 0.5*x1**2 + 2))
        g[1] = -1.0*x1*(-np.cos(12*np.sqrt(x0**2 + x1**2)) - 1)/(0.5*x0**2 + 0.5*x1**2 + 2)**2 + 12*x1*np.sin(12*np.sqrt(x0**2 + x1**2))/(np.sqrt(x0**2 + x1**2)*(0.5*x0**2 + 0.5*x1**2 + 2))
        return g
        
    def hess_vec(self, x, v):
        x0, x1 = x
        v0, v1 = v
        hv = np.zeros(2, dtype=np.float64)
        hv[0] = v0*(2.0*x0**2*(-np.cos(12*np.sqrt(x0**2 + x1**2)) - 1)/(0.5*x0**2 + 0.5*x1**2 + 2)**3 + 144*x0**2*np.cos(12*np.sqrt(x0**2 + x1**2))/((x0**2 + x1**2)*(0.5*x0**2 + 0.5*x1**2 + 2)) - 24.0*x0**2*np.sin(12*np.sqrt(x0**2 + x1**2))/(np.sqrt(x0**2 + x1**2)*(0.5*x0**2 + 0.5*x1**2 + 2)**2) - 12*x0**2*np.sin(12*np.sqrt(x0**2 + x1**2))/((x0**2 + x1**2)**(3/2)*(0.5*x0**2 + 0.5*x1**2 + 2)) - 1.0*(-np.cos(12*np.sqrt(x0**2 + x1**2)) - 1)/(0.5*x0**2 + 0.5*x1**2 + 2)**2 + 12*np.sin(12*np.sqrt(x0**2 + x1**2))/(np.sqrt(x0**2 + x1**2)*(0.5*x0**2 + 0.5*x1**2 + 2))) + v1*(2.0*x0*x1*(-np.cos(12*np.sqrt(x0**2 + x1**2)) - 1)/(0.5*x0**2 + 0.5*x1**2 + 2)**3 + 144*x0*x1*np.cos(12*np.sqrt(x0**2 + x1**2))/((x0**2 + x1**2)*(0.5*x0**2 + 0.5*x1**2 + 2)) - 24.0*x0*x1*np.sin(12*np.sqrt(x0**2 + x1**2))/(np.sqrt(x0**2 + x1**2)*(0.5*x0**2 + 0.5*x1**2 + 2)**2) - 12*x0*x1*np.sin(12*np.sqrt(x0**2 + x1**2))/((x0**2 + x1**2)**(3/2)*(0.5*x0**2 + 0.5*x1**2 + 2)))
        hv[1] = v0*(2.0*x0*x1*(-np.cos(12*np.sqrt(x0**2 + x1**2)) - 1)/(0.5*x0**2 + 0.5*x1**2 + 2)**3 + 144*x0*x1*np.cos(12*np.sqrt(x0**2 + x1**2))/((x0**2 + x1**2)*(0.5*x0**2 + 0.5*x1**2 + 2)) - 24.0*x0*x1*np.sin(12*np.sqrt(x0**2 + x1**2))/(np.sqrt(x0**2 + x1**2)*(0.5*x0**2 + 0.5*x1**2 + 2)**2) - 12*x0*x1*np.sin(12*np.sqrt(x0**2 + x1**2))/((x0**2 + x1**2)**(3/2)*(0.5*x0**2 + 0.5*x1**2 + 2))) + v1*(2.0*x1**2*(-np.cos(12*np.sqrt(x0**2 + x1**2)) - 1)/(0.5*x0**2 + 0.5*x1**2 + 2)**3 + 144*x1**2*np.cos(12*np.sqrt(x0**2 + x1**2))/((x0**2 + x1**2)*(0.5*x0**2 + 0.5*x1**2 + 2)) - 24.0*x1**2*np.sin(12*np.sqrt(x0**2 + x1**2))/(np.sqrt(x0**2 + x1**2)*(0.5*x0**2 + 0.5*x1**2 + 2)**2) - 12*x1**2*np.sin(12*np.sqrt(x0**2 + x1**2))/((x0**2 + x1**2)**(3/2)*(0.5*x0**2 + 0.5*x1**2 + 2)) - 1.0*(-np.cos(12*np.sqrt(x0**2 + x1**2)) - 1)/(0.5*x0**2 + 0.5*x1**2 + 2)**2 + 12*np.sin(12*np.sqrt(x0**2 + x1**2))/(np.sqrt(x0**2 + x1**2)*(0.5*x0**2 + 0.5*x1**2 + 2)))
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["DROPWAVE"] = DropwaveObjective


class EasomObjective:
    name = "EASOM"
    n = 2
    
    @property
    def x_init(self):
        return np.array([2.0, 2.0], dtype=np.float64)
        
    def f(self, x):
        x0, x1 = x
        return -np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.cos(x0)*np.cos(x1)
        
    def grad(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=np.float64)
        g[0] = -(-2*x0 + 2*np.pi)*np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.cos(x0)*np.cos(x1) + np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.sin(x0)*np.cos(x1)
        g[1] = -(-2*x1 + 2*np.pi)*np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.cos(x0)*np.cos(x1) + np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.sin(x1)*np.cos(x0)
        return g
        
    def hess_vec(self, x, v):
        x0, x1 = x
        v0, v1 = v
        hv = np.zeros(2, dtype=np.float64)
        hv[0] = v0*(-(-2*x0 + 2*np.pi)**2*np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.cos(x0)*np.cos(x1) + 2*(-2*x0 + 2*np.pi)*np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.sin(x0)*np.cos(x1) + 3*np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.cos(x0)*np.cos(x1)) + v1*(-(-2*x0 + 2*np.pi)*(-2*x1 + 2*np.pi)*np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.cos(x0)*np.cos(x1) + (-2*x0 + 2*np.pi)*np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.sin(x1)*np.cos(x0) + (-2*x1 + 2*np.pi)*np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.sin(x0)*np.cos(x1) - np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.sin(x0)*np.sin(x1))
        hv[1] = v0*(-(-2*x0 + 2*np.pi)*(-2*x1 + 2*np.pi)*np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.cos(x0)*np.cos(x1) + (-2*x0 + 2*np.pi)*np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.sin(x1)*np.cos(x0) + (-2*x1 + 2*np.pi)*np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.sin(x0)*np.cos(x1) - np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.sin(x0)*np.sin(x1)) + v1*(-(-2*x1 + 2*np.pi)**2*np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.cos(x0)*np.cos(x1) + 2*(-2*x1 + 2*np.pi)*np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.sin(x1)*np.cos(x0) + 3*np.exp(-(x0 - np.pi)**2 - (x1 - np.pi)**2)*np.cos(x0)*np.cos(x1))
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["EASOM"] = EasomObjective


class Schwefel5Objective:
    name = "SCHWEFEL5"
    n = 5
    
    @property
    def x_init(self):
        return np.array([420.9687, 420.9687, 420.9687, 420.9687, 420.9687], dtype=np.float64)
        
    def f(self, x):
        return -np.sum(x * np.sin(np.sqrt(np.abs(x)))) + 418.9829 * 5
        
    def grad(self, x):
        u = np.sqrt(np.abs(x))
        return -np.sin(u) - 0.5 * u * np.cos(u)
        
    def hess_vec(self, x, v):
        u = np.sqrt(np.abs(x))
        inv_u = np.zeros_like(u)
        mask = u > 1e-12
        inv_u[mask] = 1.0 / u[mask]
        diag = np.sign(x) * (0.25 * np.sin(u) - 0.75 * inv_u * np.cos(u))
        return diag * v
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["SCHWEFEL5"] = Schwefel5Objective


class GoldsteinObjective:
    name = "GOLDSTEIN"
    n = 2
    
    @property
    def x_init(self):
        return np.array([-0.5, 0.25], dtype=np.float64)
        
    def f(self, x):
        x0, x1 = x
        return ((2*x0 - 3*x1)**2*(12*x0**2 - 36*x0*x1 - 32*x0 + 27*x1**2 + 48*x1 + 18) + 30)*((x0 + x1 + 1)**2*(3*x0**2 + 6*x0*x1 - 14*x0 + 3*x1**2 - 14*x1 + 19) + 1)
        
    def grad(self, x):
        x0, x1 = x
        g = np.zeros(2, dtype=np.float64)
        g[0] = ((2*x0 - 3*x1)**2*(24*x0 - 36*x1 - 32) + (8*x0 - 12*x1)*(12*x0**2 - 36*x0*x1 - 32*x0 + 27*x1**2 + 48*x1 + 18))*((x0 + x1 + 1)**2*(3*x0**2 + 6*x0*x1 - 14*x0 + 3*x1**2 - 14*x1 + 19) + 1) + ((2*x0 - 3*x1)**2*(12*x0**2 - 36*x0*x1 - 32*x0 + 27*x1**2 + 48*x1 + 18) + 30)*((x0 + x1 + 1)**2*(6*x0 + 6*x1 - 14) + (2*x0 + 2*x1 + 2)*(3*x0**2 + 6*x0*x1 - 14*x0 + 3*x1**2 - 14*x1 + 19))
        g[1] = ((-12*x0 + 18*x1)*(12*x0**2 - 36*x0*x1 - 32*x0 + 27*x1**2 + 48*x1 + 18) + (2*x0 - 3*x1)**2*(-36*x0 + 54*x1 + 48))*((x0 + x1 + 1)**2*(3*x0**2 + 6*x0*x1 - 14*x0 + 3*x1**2 - 14*x1 + 19) + 1) + ((2*x0 - 3*x1)**2*(12*x0**2 - 36*x0*x1 - 32*x0 + 27*x1**2 + 48*x1 + 18) + 30)*((x0 + x1 + 1)**2*(6*x0 + 6*x1 - 14) + (2*x0 + 2*x1 + 2)*(3*x0**2 + 6*x0*x1 - 14*x0 + 3*x1**2 - 14*x1 + 19))
        return g
        
    def hess_vec(self, x, v):
        x0, x1 = x
        v0, v1 = v
        hv = np.zeros(2, dtype=np.float64)
        hv[0] = v0*(2*((2*x0 - 3*x1)**2*(24*x0 - 36*x1 - 32) + (8*x0 - 12*x1)*(12*x0**2 - 36*x0*x1 - 32*x0 + 27*x1**2 + 48*x1 + 18))*((x0 + x1 + 1)**2*(6*x0 + 6*x1 - 14) + (2*x0 + 2*x1 + 2)*(3*x0**2 + 6*x0*x1 - 14*x0 + 3*x1**2 - 14*x1 + 19)) + ((2*x0 - 3*x1)**2*(12*x0**2 - 36*x0*x1 - 32*x0 + 27*x1**2 + 48*x1 + 18) + 30)*(6*x0**2 + 12*x0*x1 - 28*x0 + 6*x1**2 - 28*x1 + 6*(x0 + x1 + 1)**2 + 2*(2*x0 + 2*x1 + 2)*(6*x0 + 6*x1 - 14) + 38) + ((x0 + x1 + 1)**2*(3*x0**2 + 6*x0*x1 - 14*x0 + 3*x1**2 - 14*x1 + 19) + 1)*(96*x0**2 - 288*x0*x1 - 256*x0 + 216*x1**2 + 384*x1 + 24*(2*x0 - 3*x1)**2 + 2*(8*x0 - 12*x1)*(24*x0 - 36*x1 - 32) + 144)) + v1*(((-12*x0 + 18*x1)*(12*x0**2 - 36*x0*x1 - 32*x0 + 27*x1**2 + 48*x1 + 18) + (2*x0 - 3*x1)**2*(-36*x0 + 54*x1 + 48))*((x0 + x1 + 1)**2*(6*x0 + 6*x1 - 14) + (2*x0 + 2*x1 + 2)*(3*x0**2 + 6*x0*x1 - 14*x0 + 3*x1**2 - 14*x1 + 19)) + ((2*x0 - 3*x1)**2*(24*x0 - 36*x1 - 32) + (8*x0 - 12*x1)*(12*x0**2 - 36*x0*x1 - 32*x0 + 27*x1**2 + 48*x1 + 18))*((x0 + x1 + 1)**2*(6*x0 + 6*x1 - 14) + (2*x0 + 2*x1 + 2)*(3*x0**2 + 6*x0*x1 - 14*x0 + 3*x1**2 - 14*x1 + 19)) + ((2*x0 - 3*x1)**2*(12*x0**2 - 36*x0*x1 - 32*x0 + 27*x1**2 + 48*x1 + 18) + 30)*(6*x0**2 + 12*x0*x1 - 28*x0 + 6*x1**2 - 28*x1 + 6*(x0 + x1 + 1)**2 + 2*(2*x0 + 2*x1 + 2)*(6*x0 + 6*x1 - 14) + 38) + ((x0 + x1 + 1)**2*(3*x0**2 + 6*x0*x1 - 14*x0 + 3*x1**2 - 14*x1 + 19) + 1)*(-144*x0**2 + 432*x0*x1 + 384*x0 - 324*x1**2 - 576*x1 + (-12*x0 + 18*x1)*(24*x0 - 36*x1 - 32) - 36*(2*x0 - 3*x1)**2 + (8*x0 - 12*x1)*(-36*x0 + 54*x1 + 48) - 216))
        hv[1] = v0*(((-12*x0 + 18*x1)*(12*x0**2 - 36*x0*x1 - 32*x0 + 27*x1**2 + 48*x1 + 18) + (2*x0 - 3*x1)**2*(-36*x0 + 54*x1 + 48))*((x0 + x1 + 1)**2*(6*x0 + 6*x1 - 14) + (2*x0 + 2*x1 + 2)*(3*x0**2 + 6*x0*x1 - 14*x0 + 3*x1**2 - 14*x1 + 19)) + ((2*x0 - 3*x1)**2*(24*x0 - 36*x1 - 32) + (8*x0 - 12*x1)*(12*x0**2 - 36*x0*x1 - 32*x0 + 27*x1**2 + 48*x1 + 18))*((x0 + x1 + 1)**2*(6*x0 + 6*x1 - 14) + (2*x0 + 2*x1 + 2)*(3*x0**2 + 6*x0*x1 - 14*x0 + 3*x1**2 - 14*x1 + 19)) + ((2*x0 - 3*x1)**2*(12*x0**2 - 36*x0*x1 - 32*x0 + 27*x1**2 + 48*x1 + 18) + 30)*(6*x0**2 + 12*x0*x1 - 28*x0 + 6*x1**2 - 28*x1 + 6*(x0 + x1 + 1)**2 + 2*(2*x0 + 2*x1 + 2)*(6*x0 + 6*x1 - 14) + 38) + ((x0 + x1 + 1)**2*(3*x0**2 + 6*x0*x1 - 14*x0 + 3*x1**2 - 14*x1 + 19) + 1)*(-144*x0**2 + 432*x0*x1 + 384*x0 - 324*x1**2 - 576*x1 + (-12*x0 + 18*x1)*(24*x0 - 36*x1 - 32) - 36*(2*x0 - 3*x1)**2 + (8*x0 - 12*x1)*(-36*x0 + 54*x1 + 48) - 216)) + v1*(2*((-12*x0 + 18*x1)*(12*x0**2 - 36*x0*x1 - 32*x0 + 27*x1**2 + 48*x1 + 18) + (2*x0 - 3*x1)**2*(-36*x0 + 54*x1 + 48))*((x0 + x1 + 1)**2*(6*x0 + 6*x1 - 14) + (2*x0 + 2*x1 + 2)*(3*x0**2 + 6*x0*x1 - 14*x0 + 3*x1**2 - 14*x1 + 19)) + ((2*x0 - 3*x1)**2*(12*x0**2 - 36*x0*x1 - 32*x0 + 27*x1**2 + 48*x1 + 18) + 30)*(6*x0**2 + 12*x0*x1 - 28*x0 + 6*x1**2 - 28*x1 + 6*(x0 + x1 + 1)**2 + 2*(2*x0 + 2*x1 + 2)*(6*x0 + 6*x1 - 14) + 38) + ((x0 + x1 + 1)**2*(3*x0**2 + 6*x0*x1 - 14*x0 + 3*x1**2 - 14*x1 + 19) + 1)*(216*x0**2 - 648*x0*x1 - 576*x0 + 486*x1**2 + 864*x1 + 2*(-12*x0 + 18*x1)*(-36*x0 + 54*x1 + 48) + 54*(2*x0 - 3*x1)**2 + 324))
        return hv
        
    def hvp(self, x, v):
        return self.hess_vec(x, v)
PROBLEMS["GOLDSTEIN"] = GoldsteinObjective

