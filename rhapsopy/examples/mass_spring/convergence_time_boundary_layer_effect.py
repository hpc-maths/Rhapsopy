#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:47:24 2023

@author: lfrancoi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import trapz as trapezoid
from scipy.integrate import cumtrapz as cumulative_trapezoid
import scipy.interpolate
from rhapsopy.coupling import Orchestrator, BaseCoupler
from rhapsopy.examples.mass_spring.model import SpringMassCoupler
from time_boundary_layer_effect import performCoupledSimulation, tf
from time_boundary_layer_effect import sol_fully_coupled as solref
from tqdm import tqdm

bExplicitCoupling = False
order = 5
# nt_vec = np.unique( np.logspace(np.log10(max(5,order+1)), np.log10(100), 10).astype(int) )
nt_vec = np.unique( np.logspace(np.log10(max(5,order+1)), np.log10(1000), 10).astype(int) )

sols = []

# sol_names=['no init', 'pre init', 'progressive dt', 'post init', 'post init WR', 'post init WR2', ]

sol_names=['no init', 'pre init', 'post init',  'progressive dt', 'post init WR']
for i in range(order):
    sol_names.append(f'new post init WR N-{i}')

for nt in tqdm(nt_vec):
    sols.append([])

    sol_noinit,   pred_noinit   = performCoupledSimulation(nt=nt, order=order, mode='noinit', bExplicitCoupling=bExplicitCoupling)
    sol_preinit,  pred_preinit  = performCoupledSimulation(nt=nt, order=order, mode='preinit', bExplicitCoupling=bExplicitCoupling)
    sol_postinit, pred_postinit = performCoupledSimulation(nt=nt, order=order, mode='postinit', bExplicitCoupling=bExplicitCoupling)
    # sol_boundary, pred_boundary = performCoupledSimulation(nt=nt, order=order, first_dt_factor=1e-3, growth=1.5, mode='noinit', bExplicitCoupling=bExplicitCoupling)
    sol_boundary, pred_boundary = performCoupledSimulation(nt=nt, order=order, first_dt_factor=1e-2, growth=3., mode='noinit', bExplicitCoupling=bExplicitCoupling)
    sol_postinit_iter, pred_postinit_iter = performCoupledSimulation(nt=nt, order=order, mode='postinit_iterative', bExplicitCoupling=bExplicitCoupling)
    sols[-1].append( sol_noinit )
    sols[-1].append( sol_preinit )
    sols[-1].append( sol_boundary )
    sols[-1].append( sol_postinit )
    sols[-1].append( sol_postinit_iter )

    for i in range(order): # test high-order initialisation with various time step counts
      sol_postinit_iter2, pred_postinit_iter2 = performCoupledSimulation(nt=nt, order=order, mode=f'postinit_iterative_new_nm{i}', bExplicitCoupling=bExplicitCoupling)
      sols[-1].append( sol_postinit_iter2 )

#%% Plot finest solutions
plt.figure()
plt.plot(solref.t, solref.y[0,:], label='ref')
plt.plot(sols[-1][-1].t, sols[-1][-1].y[0,:], label='finest')
plt.grid()
plt.xlabel('t (s)')
plt.ylabel('var 0')

#%% Errors
err1=np.zeros((len(sols[0]), nt_vec.size))
err2=np.zeros_like(err1)

for i, current_nt in enumerate(nt_vec):
     current_sols = sols[i]
     for isol, current_sol in enumerate(current_sols):
         nt = current_sol.t.size

         interpolated_reference_sol = solref.sol(current_sol.t) # interpolation de la solution de référence sur la même grille temporelle
         current_sol.err = current_sol.y - interpolated_reference_sol
         current_sol.cumul_err = cumulative_trapezoid( abs(current_sol.y - interpolated_reference_sol),
                                       current_sol.t, axis=1)
         # ceci correspond à l'erreur sur la solution au temps final
         err1[isol,i]=abs(current_sol.y[0,-1] - interpolated_reference_sol[0,-1]) # error at final time
         # this may yield superconvergence is tf matches an extrema of the first variable
         # err1[isol,i] = np.linalg.norm(current_sol.y[:,-1] - solref.y[:,-1])/np.sqrt(solref.y.shape[0]) # error at final time
         # err1[isol,i] = np.linalg.norm(current_sol.y[:,-1] - interpolated_reference_sol[:,-1])/np.sqrt(interpolated_reference_sol.shape[0]) # error at final time

         # L1-norm of the error in time compared to the reference solution

         err_all_vars = trapezoid( abs(current_sol.y - interpolated_reference_sol),
                                       current_sol.t, axis=1)
         # = current_sol.cumul_err[-1]
         err2[isol,i] = (1/tf) * np.sum(err_all_vars, axis=0) / err_all_vars.shape[0]


#%% Plots

dt_vec = tf/nt_vec
# Numerical estimation of convergence orders
for err, name in ((err1, 'Error at final time'),
                  (err2, 'L1 time error')):
    plt.figure()
    for isol, solname in enumerate(sol_names):
        plt.loglog(dt_vec, err[isol,:],label=solname, marker='.')
    plt.grid()
    plt.legend()
    plt.xlabel("dt")
    plt.ylabel("error")
    plt.title(name)

    plt.figure()
    for isol, solname in enumerate(sol_names):
        plt.semilogx(dt_vec, np.gradient(np.log10(err2[isol,:]), np.log10(dt_vec)), label=solname)
    plt.grid()
    plt.ylim(0,order+2)
    plt.legend()
    plt.xlabel(r"$\Delta t$ (s)")
    plt.ylabel("order")
    plt.title(name)

#%% Plot error accumulation
for isol, solname in enumerate(sol_names):
    plt.figure()

    for i, current_nt in enumerate(nt_vec):
         nt = current_sol.t.size
         current_sol = sols[i][isol]
         plt.semilogy(current_sol.t, abs(current_sol.err[0,:]), label=f'nt={nt}')
    plt.legend(ncol=4, loc='lower center', framealpha=0.4)
    plt.xlabel('t')
    plt.ylabel('y-yref')
    plt.grid()
    plt.title(f'Error for {solname}')
    plt.ylim(1e-14,1e0)
    plt.xlim(0,tf)


