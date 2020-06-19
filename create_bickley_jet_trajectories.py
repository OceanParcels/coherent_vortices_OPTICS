"""
"Ordering of trajectories reveals hierarchical finite-time coherent sets in Lagrangian particle data"
David Wichmann, Christian Kehl, Henk A. Dijkstra, and Erik van Sebille
questions to d.wichmann@uu.nl

Create bickley-jet trajectories.
"""

import numpy as np
from scipy.integrate import odeint

#Parameters for the Bickley jet
U = 0.06266
L = 1770.
r0 = 6371.
k1 = 2 * 1/ r0
k2 = 2 * 2 / r0
k3 = 2 * 3/ r0
eps1 = 0.075
eps2 = 0.4
eps3 = 0.3
c3 = 0.461 * U
c2 = 0.205 * U
c1 = 0.1446 * U

def vel(y,t): #2D velocity
    x1 = y[0]
    x2 = y[1]   
    f1 = eps1 * np.exp(-1j *k1 * c1 * t)
    f2 = eps2 * np.exp(-1j *k2 * c2 * t)
    f3 = eps3 * np.exp(-1j *k3 * c3 * t)
    F1 = f1 * np.exp(1j * k1 * x1)
    F2 = f2 * np.exp(1j * k2 * x1)
    F3 = f3 * np.exp(1j * k3 * x1)    
    G = np.real(np.sum([F1,F2,F3]))
    G_x = np.real(np.sum([1j * k1 *F1, 1j * k2 * F2, 1j * k3 * F3]))    
    u =  U / (np.cosh(x2/L)**2)  +  2 * U * np.sinh(x2/L) / (np.cosh(x2/L)**3) *  G
    v = U * L * (1./np.cosh(x2/L))**2 * G_x    
    return [u,v]
    
tau = 40 * 86400 #days in seconds
dt_output = 86400/10 #output every 0.1 days
dt_int = 1
n_it = int(dt_output / dt_int) #index of output

#discretization
t = np.arange(0, tau + dt_output, dt_int) 
x_range = np.linspace(0,1,201) * np.pi * r0 #in km
x_range = x_range[:-1]
y_range = np.linspace(-3.,3.,60) * 1000 #in km

#initial conditions
X0, Y0 = np.meshgrid(x_range, y_range)
X0 = X0.flatten()
Y0 = Y0.flatten()

#empty trajectory arrays
trajectories_x = np.empty((X0.shape[0],int(tau/dt_output)+1))
trajectories_y = np.empty((Y0.shape[0],int(tau/dt_output)+1))

#integrate trajectories
for i in range(len(X0)):
    if i % 50 == 0: print(str(i) + " / " + str(len(X0)))
    x0 = X0[i]
    y0 = Y0[i]
    sol = odeint(vel, [x0,y0], t)
    trajectories_x[i] = sol[:,0][::n_it] % (np.pi * r0)
    trajectories_y[i] = sol[:,1][::n_it]

np.savez("bickley_jet_trajectories", drifter_longitudes = trajectories_x, 
          drifter_latitudes = trajectories_y, drifter_time = [])
