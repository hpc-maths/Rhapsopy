#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:02:47 2021

Test the adaptive code coupling procedure on the heat conduction case

Visualise the time step evolution, step rejection

@author: laurent
"""
import numpy as np
np.seterr(divide="raise")
import scipy.integrate
import matplotlib.pyplot as plt

from rhapsopy.examples.heat1D.model.heat import getTs, getEnergy
from rhapsopy.examples.heat1D.model.setup_heat import getSetup
from rhapsopy.examples.heat1D.model.coupler_heat import Coupler
from rhapsopy.coupling import Orchestrator
from rhapsopy.accelerators import NewtonSolver, DampedNewtonSolver, JFNKSolver, IQNSolver, AitkenUnderrelaxationSolver, FixedPointSolver, ExplicitSolver

# N=30
# max_nt = 400
# bReactive = True
# rho1_sur_rho2 = 50.

N=50
max_nt = 4000
bReactive = False
rho1_sur_rho2 = 1/100.

tf = 1e3 # physical time simulated
# coupling_modes = ('neumann_extrap', 'dirichlet_extrap') # le plus classique
# coupling_modes = ('balance', 'neumann_extrap') # comme CEDRE
# coupling_modes = ('dirichlet_extrap', 'neumann_extrap')
# coupling_modes = ('dirichlet_extrap', 'dirichlet')
# coupling_modes = ('dirichlet', 'neumann_extrap')
coupling_modes = ('dirichlet', 'dirichlet')
# coupling_modes = ('dirichlet', 'neumann')
# coupling_modes = ('neumann', 'dirichlet')
# coupling_modes = ('neumann', 'neumann')

# error estimation
bWR = True
order = 4; rtol = 1e-3

keepUniformSampling = False
embedded_method = True
higher_order_embedded = True

first_step = 1e-4
rtol_ref = 1e-11
t_span = [0, tf] 


#%%
options1, options2 = getSetup(N1=N+1,N2=N+1, rho1_sur_rho2=rho1_sur_rho2, bReactive=bReactive)
y0_global = np.hstack( (options1['y0'], options2['y0']) )
x_global =  np.hstack( (options1['solid']['mesh']['cellX'], options2['solid']['mesh']['cellX']) )
nx1 = options1['y0'].size
nx2 = options2['y0'].size

coupler = Coupler(options1, options2, coupling_modes=coupling_modes)
coupler.adaptive_subsolves = True
coupler.method = 'Radau'

conductor = Orchestrator(coupler=coupler, order=order)

conductor.embedded_method = embedded_method
conductor.higher_order_embedded = higher_order_embedded

conductor.logger.setLevel(30)
conductor.solverlogger.setLevel(100)
coupler.logger.setLevel(100)

if bWR: # implicit coupling
  # conductor.interfaceSolver = FixedPointSolver;            conductor.NITER_MAX = 100
  # conductor.interfaceSolver = AitkenUnderrelaxationSolver; conductor.NITER_MAX = 100
  # conductor.interfaceSolver = NewtonSolver;                conductor.NITER_MAX = 15
  conductor.interfaceSolver = DampedNewtonSolver;          conductor.NITER_MAX = 50
  # conductor.interfaceSolver = JFNKSolver;          conductor.NITER_MAX = 50
else: # explicit coupling
  conductor.interfaceSolver = ExplicitSolver
  conductor.waveform_tolerance = 1e50
  conductor.raise_error_on_non_convergence = False



#%% Code coupling
outCoupled = conductor.adaptive_integration(y0=y0_global,
                                            t_span=np.array(t_span),
                                            first_step=first_step,
                                            atol=rtol, rtol=rtol,
                                            max_nt=max_nt,
                                            keepUniformSampling=keepUniformSampling)

outCoupled = coupler.complementOutput(outCoupled) # compute additional variables as post-processing

# count total number of subsystem integration calls
nWR = sum(outCoupled.WR_iters)
try:
    nWR2 = sum(outCoupled.WR_iters2)
    nWR = nWR + nWR2
except:
    pass
print('\n=======\n\nCoupled solution obtained in {} s, {} steps ({} accepted, {} rejected, {} failed), {} WR iterations'.format(
        outCoupled.CPUtime, outCoupled.nsteps_total, outCoupled.nsteps_accepted, outCoupled.nsteps_rejected, outCoupled.nsteps_failed, nWR))

#%% Ref solution
outRef = scipy.integrate.solve_ivp( fun=lambda t,y: coupler.coupledODE(t=t,y=y),
                                    y0=y0_global, method='LSODA', max_step=np.inf,
                                    t_span=[outCoupled.t[0], outCoupled.t[-1]],
                                    atol=rtol_ref, rtol=rtol_ref,
                                    uband=3, lband=3,
                                    vectorized=False, args=(), jac=None,
                                    dense_output=True)
outRef = coupler.complementOutput(outRef) # compute additional variables as post-processing
print(f'Reference solution obtained in {outRef.nfev} fev')
print('  --> "{}"'.format(outRef.message))
if not outRef.success:
    raise Exception('Reference integration failed')

  
#%%
plt.figure()
plt.plot(outCoupled.t, outCoupled.Ts, label='coupled')
plt.plot(outRef.t, outRef.Ts, label='ref')
plt.legend()
plt.xlabel('t (s)')
plt.ylabel(r'$T_s$ (K)')
plt.grid()
plt.title('Evolution of the surface temperature')
plt.xlim(0,10)


#%% Process detailed step info
from rhapsopy.coupling import process_step_info
step_info = process_step_info(outCoupled)

#%% Analyse time step adaptation
bPlotOrders=False # plot order evolution
bPlotWR=True # plot WR iterations
bPlotRef=True # plot reference result
bPlotStepInfo=True # plot failed steps and others
bTimeAxis=False # plot versus time or index

if bTimeAxis:
  Xdata = outCoupled.t
  Xdataref = outRef.t
  key_infox = 'tn'
else:
  Xdata = np.array(range(outCoupled.t.size))
  # Xdataref = np.array(range(outRef.t.size))
  Xdataref = scipy.interpolate.interp1d(x=outCoupled.t, y=Xdata, kind='linear')( outRef.t ) # sort of equivalence with the coupled solution (based on outCoupled.t)
  # Xdataref = scipy.interpolate.interp1d(x=Ts, y=Xdata, kind='linear', fill_value="extrapolate")( Tsref ) # sort of equivalence with the coupled solution (based on Ts)
  key_infox = 'nt'
  for key in step_info.keys():
    step_info[key]['nt'] = np.array( [ np.argmin(abs(outCoupled.t-t)) for t in step_info[key]['tn'] ] )
    

fig, ax = plt.subplots(2+1*bPlotWR+1*bPlotOrders, 1, sharex=True, dpi=300)

ix=0
ax[ix].plot(Xdata, outCoupled.Ts, marker='+', color='tab:blue', label='coupled')
if bPlotRef:
  ax[ix].plot(Xdataref, outRef.Ts, marker=None, color='tab:red', label='ref')
ax[ix].set_ylabel(r'$T_s$')

ix+=1
ax[ix].semilogy(Xdata[:-1], np.diff(outCoupled.t), marker='+', color='tab:blue')
if bPlotRef:
  ax[ix].semilogy(Xdataref[:-1], np.diff(outRef.t), marker=None, color='tab:red')
if bPlotStepInfo:
  ax[ix].semilogy(step_info['ERRORTOOHIGH'][key_infox], step_info['ERRORTOOHIGH']['dt'], marker='*', linestyle='',
                          color='tab:green', label='rejected')
  ax[ix].semilogy(step_info['WRNONCONVERGENCE'][key_infox], step_info['WRNONCONVERGENCE']['dt'], marker='*', linestyle='',
                              color='tab:red', label='main WR failed')
  ax[ix].semilogy(step_info['WRNONCONVERGENCE2'][key_infox], step_info['WRNONCONVERGENCE2']['dt'], marker='*', linestyle='',
                              color='tab:orange', label='embedded WR failed')
ax[ix].set_ylabel(r'$dt$ (s)')
ax[ix].grid(which='minor')
ax[ix].legend(framealpha=0, ncol=1)

if bPlotOrders:
  ix+=1
  for i in range(  outCoupled.p_hist.shape[1] ):
    ax[ix].plot(Xdata[:-1], outCoupled.p_hist[:,i], marker='+', label=f'var {i}')
  ax[ix].set_ylabel(r'order')
  ax[ix].legend()
  ax[ix].set_ylim(0, order+1)
  ax[ix].set_yticks(range(order+1), minor=True)
  ax[ix].yaxis.grid(which='minor')

if bPlotWR:
  ix+=1
  ax[ix].plot(Xdata[:-1], outCoupled.WR_iters, marker='+', color='tab:blue', label='main')
  try:
      ax[ix].plot(Xdata[:-1], outCoupled.WR_iters2, marker='+', color='tab:green', label='embedded')
  except (AttributeError, ValueError):
      pass
  ax[ix].legend()
  ax[ix].set_ylabel(r'WR iters')
  ax[ix].set_ylim(0,None)

for ix in range(len(ax)):
  ax[ix].grid()
ax[0].legend()
fig.suptitle(f'Adaptive code coupling (order {conductor.order}, rtol={rtol:.1e})')

if bTimeAxis:
  plt.xlim(outCoupled.t[0], outCoupled.t[-1])


#%% Plot temperature history
plt.figure(dpi=300)
for i in range(0, len(outCoupled.t)-1, len(outCoupled.t)//5):
    current_t = outCoupled.t[i]
    p, = plt.plot(x_global, outCoupled.y[:,i], marker=None,
                  label='t={:.2e} s'.format(current_t))
    plt.plot(x_global, outRef.sol(current_t), marker=None,
             linestyle='--', linewidth=3,
             color=p.get_color(), label=None)
# hack legend
plt.plot(np.nan, np.nan, linestyle='-', linewidth=1, color=[0,0,0], label='coupling')
plt.plot(np.nan, np.nan, linestyle='--', linewidth=3, color=[0,0,0], label='ref')
plt.grid()
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$T$')
plt.title('Evolution of the temperature profile in both slabs');
