#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:02:47 2021

Test thefully-coupled solution of the 1D conjugate heat transfer problem

@author: laurent
"""
import numpy as np
np.seterr(divide="raise")
import scipy.integrate
import matplotlib.pyplot as plt
from rhapsopy.examples.heat1D.model.setup_heat import getSetup
from rhapsopy.examples.heat1D.model.coupler_heat import Coupler


# N=50
# bReactive = False
# rho1 = 1.;   rho2=100.
# # rho1 = 100.; rho2=1.
# rho1_sur_rho2 = rho1/rho2
# t_span = [0, 1e3]

N=50
bReactive = True
rho1 = 1.;   rho2=100.
# rho1 = 100.; rho2=1.
rho1_sur_rho2 = rho1/rho2
t_span = [0, 1e3]


rtol_ref = 1e-11


#%%
options1, options2 = getSetup(N1=N+1,N2=N+1, rho1_sur_rho2=rho1_sur_rho2, rho2=rho2, bReactive=bReactive)
y0_global = np.hstack( (options1['y0'], options2['y0']) )
x_global =  np.hstack( (options1['solid']['mesh']['cellX'], options2['solid']['mesh']['cellX']) )
nx1 = options1['y0'].size
nx2 = options2['y0'].size

# coupling_modes = ('neumann_extrap', 'dirichlet_extrap') # le plus classique
# coupling_modes = ('balance', 'neumann_extrap') # comme CEDRE
# coupling_modes = ('dirichlet_extrap', 'neumann_extrap')
# coupling_modes = ('dirichlet_extrap', 'dirichlet')
# coupling_modes = ('dirichlet', 'neumann_extrap')
coupling_modes = ('dirichlet', 'dirichlet')
# coupling_modes = ('dirichlet', 'neumann')
# coupling_modes = ('neumann', 'dirichlet')
# coupling_modes = ('neumann', 'neumann')
coupler = Coupler(options1, options2, coupling_modes=coupling_modes)


#%% Ref solution
out = scipy.integrate.solve_ivp(fun=lambda t,y: coupler.coupledODE(t=t,y=y),
                                    y0=y0_global, method='LSODA', max_step=np.inf,
                                    t_span=t_span,
                                    atol=rtol_ref, rtol=rtol_ref,
                                    uband=3, lband=3,
                                    vectorized=False, args=(), jac=None,
                                    dense_output=True)

out = coupler.complementOutput(out)
print(f'Reference solution obtained in {out.t.size} steps, {out.nfev} fev')
print('  --> "{}"'.format(out.message))
if not out.success:
    raise Exception('Reference integration failed')

ref_couplingvars = []
for t,y in zip(out.t, out.y.T):
  ref_couplingvars.append( coupler.getCouplingVars(t, y) )
out.z = np.vstack( ref_couplingvars ).T

#%%
plt.figure()
plt.plot(out.t, out.Ts)
plt.xlabel('t (s)')
plt.ylabel(r'$T_s$ (K)')
plt.grid()
plt.title('Evolution of the surface temperature')
plt.xlim(0,20)

#%%
plt.figure()
plt.semilogx(out.t, out.Ts)
plt.legend()
plt.xlabel('t (s)')
plt.ylabel(r'$T_s$ (K)')
plt.grid()
plt.title('Evolution of the surface temperature')
plt.xlim(1e-3,1e3)

#%% Plot temperature history of the reference
plt.figure(dpi=300)
t_plot = np.hstack((0.,np.logspace(-3, np.log10(t_span[-1]), 10)))

for i,t in enumerate(t_plot):
  # sol = out.y[:,i]
  sol = out.sol(t)
  plt.plot( np.sign(np.log(rho1_sur_rho2)) * x_global, sol, marker='+', label='t={:.2e} s'.format(t))
plt.grid()
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$T$')
plt.title(f'Evolution of the temperature profile in both slabs\n(rho1/rho2={rho1_sur_rho2})')


#%% energy
plt.figure()
plt.plot(out.t, out.E1+out.E2, label='total', linestyle='--')
plt.plot(out.t, out.E1, label='E1', linestyle='-')
plt.plot(out.t, out.E2, label='E2', linestyle='-')
plt.legend()
plt.grid()
plt.xlabel('t (s)')
plt.ylabel('E (J)')

#%% Energy conservation
plt.figure()
plt.plot(out.t, (out.Etot-out.Etot[0])/out.Etot[0], label='total', linestyle='--')
plt.grid()
plt.xlabel('t (s)')
plt.ylabel('$\Delta E / E_0$ (J)')
