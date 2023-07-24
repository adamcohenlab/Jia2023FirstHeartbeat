import numpy as np
from pde import PDEBase, MemoryStorage, ScalarField, plot_kymograph, UnitGrid, FieldCollection
import matplotlib.pyplot as plt
import colorcet as cc
from pde.tools.numba import jit
from skimage import io as skio
from skimage import exposure, transform
from scipy import optimize, interpolate

import os
import sys

def prep_visualize(array, stride=1, rescale=10, dtype=np.uint8):
    downsampled_t = array[::stride]
    if rescale > 1:
        downsampled_t = transform.rescale(downsampled_t, (1,\
            rescale, rescale), preserve_range=True, anti_aliasing=False,\
                order=0)
    return exposure.rescale_intensity(downsampled_t, out_range=(0,255)).astype(dtype)

def gen_ode(E_rev=[0,0,0], n_h=0, g=[0,0,0], m_h=0,\
        k_m=0, k_n=0, tau=0, C=0, I=0, D=0, f=lambda x, t: 0):
    def ode(V, n, t):
        def m_inf(x):
            return 1/(1+np.exp(-(x-m_h)/k_m))
        def n_inf(x):
            return 1/(1+np.exp(-(x-n_h)/k_n))
        V_t = (I - g[0]*(V-E_rev[0]) - \
        g[1]*m_inf(V)*(V-E_rev[1]) - \
        g[2]*n*(V-E_rev[2]))/C + \
        f([V,n], t)
        
        n_t = (n_inf(V) - n)/tau
        return V_t, n_t
    return ode

def gen_nullclines(E_rev=[0,0,0], n_h=0, g=0, m_h=0,\
        k_m=0, k_n=0, tau=0, C=0, I=0, D=0):
    def nullV(V):
        def m_inf(x):
            return 1/(1+np.exp(-(x-m_h)/k_m))
        n_Vt = (I-g[0]*(V-E_rev[0]) - g[1]*m_inf(V)*(V-E_rev[1]))/(g[2]*(V-E_rev[2]))
        return n_Vt
    def nulln(V):
        def n_inf(x):
            return 1/(1+np.exp(-(x-n_h)/k_n))
        n_nt = n_inf(V)
        return n_nt
    return nullV, nulln

def gen_phase_function(params, bif, t_max=50, dt=0.001):
    # min_bif_values = {"snic": 4.6, "saddle_node": 4.6, "subcritical_hopf": 30, "supercritical_hopf": 8}
    initial_guesses = {"snic": [-60, -29], "saddle_node": [-60, -29], "supercritical_hopf": [-40], "subcritical_hopf": [-40]}
    params_ode = params.copy()
    params_ode["I"] = np.max(params["I"])
    # is this correct?
    # if params_ode["I"] < min_bif_values[bif]:
    #     params_ode["I"] = min_bif_values[bif]
        
    nullV, nulln = gen_nullclines(**params_ode)
    critical_points= optimize.fsolve(lambda x: nullV(x) - nulln(x), initial_guesses[bif])
    n_crit = nulln(critical_points)
    if len(critical_points) >1:
        crit = (critical_points[-1], n_crit[-1])
    else:
        crit = (critical_points[0], n_crit[0])
        
    ode = gen_ode(**params_ode)
    v_test = np.linspace(-70, - 10, 100)
    n_test = nulln(v_test)
    dVdt, _ = ode(v_test, n_test, 0)
    v_i = v_test[np.argmax(dVdt)]
    n_i = nulln(v_i)

    n_steps = int(np.round(t_max/dt))
    zs = np.zeros([n_steps,2])
    
    zs[0,:] = np.array([v_i, n_i])
    for i in range(1,n_steps):
        V_t, n_t = ode(zs[i-1,0], zs[i-1,1], i*dt)
        zs[i,:] = zs[i-1,:] + np.array([V_t, n_t])*dt
    phase = np.arctan2((zs[:,1]-crit[1]), (zs[:,0]-crit[0]))
    zero_crossings = np.argwhere((phase*np.roll(phase, 1) < 0) & \
                    (np.diff(phase, prepend=[0]) > 0)).ravel()
    ### Detect decaying oscillations
    if len(zero_crossings) < 2:
        raise Exception("Could not identify periodic function")
    else:
        single_rot_2pi = np.copy(phase[zero_crossings[-2]:zero_crossings[-1]])
        single_rot_2pi[single_rot_2pi < 0] = 2*np.pi + single_rot_2pi[single_rot_2pi < 0]
        min_val = np.min(single_rot_2pi)
        max_val = np.max(single_rot_2pi)
        print(min_val, max_val)

        phase_func = interpolate.interp1d(single_rot_2pi, np.linspace(min_val, max_val,\
                len(single_rot_2pi)), fill_value="extrapolate")
    return phase_func, crit

def solve_fixed_points(params, v0):
    params_ode = params.copy()
    params_ode["I"] = np.max(params["I"])
    nullV, nulln = gen_nullclines(**params_ode)
    ode = gen_ode(**params_ode)
    results = []
    for v in v0:
        results.append(optimize.minimize(lambda x: (nullV(x) - nulln(x))**2, v, tol=1e-7))
    
    critical_points = np.array([r.x for r in results]).ravel()
    errs = (nullV(critical_points) - nulln(critical_points))**2
    n_crit = nulln(critical_points)
    
    
    dx = 0.01
    dy = 0.001
    dfdx = (np.array(ode(critical_points+dx, n_crit, 0)) - np.array(ode(critical_points-dx, n_crit, 0)))/(2*dx)
    dfdy = (np.array(ode(critical_points, n_crit+dy, 0)) - np.array(ode(critical_points, n_crit-dy, 0)))/(2*dy)
    jacs = np.array([dfdx, dfdy])
    ls = []
    for i in range(jacs.shape[2]):
        l = np.linalg.eig(jacs[:,:,i].T)[0]
        # print(l)
        ls.append(l)
    return np.array([critical_points, nulln(critical_points)]), ls, errs

def find_max_dVdt(params, bif, t_max=100, dt = 0.001):
    # initial_guesses = {"snic": [-60, -29], "saddle_node": [-60, -29], "supercritical_hopf": [-40], "subcritical_hopf": [-40]}
    params_ode = params.copy()
    params_ode["I"] = np.max(params["I"])
    ode = gen_ode(**params_ode)
    fp, ls, errs = solve_fixed_points(params_ode, [-80, -50, -10])
    stable = np.array([np.all(l< 0) for l in ls])
    
    fp = fp[:, errs < 1e-3]
    stable = stable[errs < 1e-3]
    if np.sum(~stable) >0:
        crit = fp[:, ~stable][:,-1]
    else:
        crit = fp[:,0]
    
    v_i = crit[0] + 10
    n_i = crit[1] + 0.02
    
    n_steps = int(np.round(t_max/dt))
    zs = np.zeros([n_steps,2])
    zs[0,:] = np.array([v_i, n_i])
    for i in range(1,n_steps):
        V_t, n_t = ode(zs[i-1,0], zs[i-1,1], i*dt)
        zs[i,:] = zs[i-1,:] + np.array([V_t, n_t])*dt
    rel_zs = zs - crit
    rel_zs[:,1] *= 100
    radii = np.sum((rel_zs)**2, axis=1)**0.5
    angles = np.arctan2(rel_zs[:,1], rel_zs[:,0])
    dVdt, _ = ode(zs[:,0], zs[:,1], 0)
    
    theta = angles[np.argmax(-dVdt)]
    if theta < 0:
        theta = np.pi*2 + theta
    return theta, crit
    
def draw_phase_plane(params, xlim, ylim, figsize=(12,12), grid_size=30, quiver_scale=1000):
    ode = gen_ode(**params)
    nullV, nulln = gen_nullclines(**params)
    V = np.linspace(*xlim, grid_size)
    n = np.linspace(*ylim, grid_size)
    X, Y = np.meshgrid(V, n)
    sim_xs = np.array([X.ravel(), Y.ravel()])
    dVdt, dndt = ode(X.ravel(), Y.ravel(), 0)
    Vfine = np.linspace(*xlim, 100)

    
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.quiver(X.ravel(),Y.ravel(), dVdt, dndt, angles="xy", scale=quiver_scale)
    ax1.plot(Vfine, nullV(Vfine), color="magenta")
    ax1.plot(Vfine, nulln(Vfine), color="yellow")
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    return ax1



class INapIkPDE(PDEBase):
    def __init__(self, params, f=lambda x, t: 0,\
                 bc="auto_periodic_neumann", sigma=0):
        super().__init__(noise=[sigma,0])
        self.E_rev = params["E_rev"]
        self.n_h = params["n_h"]
        self.k_n = params["k_n"]
        self.g = params["g"]
        self.m_h = params["m_h"]
        self.k_m = params["k_m"]
        self.tau = params["tau"]
        self.C = params["C"]
        self.I = params["I"]
        self.D = params["D"]
        self.f = f
        self.bc = bc
        self.ode = gen_ode(**params)
    
    def evolution_rate(self, state, t=0):
        V, n = state
        
        V_ode, n_ode = self.ode(V, n, t)
        
        V_t = V_ode + self.D*V.laplace(bc=self.bc)
        n_t = n_ode
        return FieldCollection([V_t, n_t])
    
    def _make_pde_rhs_numba(self, state):
        E_rev = self.E_rev 
        n_h = self.n_h
        k_n = self.k_n
        g = self.g
        m_h = self.m_h
        k_m = self.k_m
        tau = self.tau
        C = self.C
        I = self.I
        D = self.D
#         f = self.f
        
        laplace = state.grid.make_operator("laplace", bc=self.bc)
#         @jit
#         def input_function(state_data, t=0):
#             return f(state_data, t)
        
        @jit(nopython=True)
        def pde_rhs(state_data, t=0):
            V = state_data[0]
            n = state_data[1]
            rate = np.empty_like(state_data)
            m_inf = 1/(1+np.exp(-(V-m_h)/k_m))
            n_inf = 1/(1+np.exp(-(V-n_h)/k_n))
            
            
            rate[0] = (I - g[0]*(V-E_rev[0]) - \
                g[1]*m_inf*(V-E_rev[1]) - \
                g[2]*n*(V-E_rev[2]))/C + \
                 + D*laplace(V)
            rate[1] = (n_inf - n)/tau
            return rate
        return pde_rhs
            
        
    def _m_inf(self, x):
        return 1/(1+np.exp(-(x-self.m_h)/self.k_m))
    
    def _n_inf(self, x):
        return 1/(1+np.exp(-(x-self.n_h)/self.k_n))

