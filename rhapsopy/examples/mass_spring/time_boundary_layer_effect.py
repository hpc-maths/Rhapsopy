#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:12:29 2023

@author: lfrancoi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import trapz as trapezoid
import scipy.interpolate
from rhapsopy.coupling import Orchestrator, BaseCoupler
from rhapsopy.examples.mass_spring.model import SpringMassCoupler
from rhapsopy.accelerators import NewtonSolver, DampedNewtonSolver, IQNSolver, AitkenUnderrelaxationSolver, AitkenScalarSolver, FixedPointSolver, AndersonSolver, ExplicitSolver

#%% Parameters
kspring=10.
mass=1.
L0=1.
d=0.66
y0_global = np.array([0.3, 0, 0.8, 0])
tf_transient = 3.0 #10.3 #initial transient for optional initialisation of predictors

order = 4 # predicition order
bExplicitCoupling = True # If False, WR iterations are performed to obtain an implicit coupling

nt = 20 # number of coupling time steps
tf = 2.0 # Physical time simulated
# nt = 100
# tf = 10.

y0_global = np.array([0.3, 0, 0.8, 0])

#%% Compute reference solution
# Compute a reference solution (**including** the initial transient in negative time)
coupler = SpringMassCoupler(kspring=kspring,mass=mass,L0=L0,d=d)
sol_fully_coupled = solve_ivp(fun=coupler.coupled_ode,
                y0=y0_global, t_span=[-tf_transient,tf+tf_transient],
                method='DOP853', atol=1e-13, rtol=1e-13,
                dense_output=True)
y0_global[:] = sol_fully_coupled.sol(0.) # the new initial state is that obtained after "tf_transient" seconds

#%% Perform coupled simulation with predictor initialisation
def performCoupledSimulation(mode, nt=nt, order=order, first_dt_factor=1, growth=1, bExplicitCoupling=True):

    if growth==1:
        if first_dt_factor!=1:
            raise Exception('No growth and first_factor!=1 is not possible')
        t_vec = np.linspace(0, tf, nt+1)
    else:
        design_dt = tf/nt
        first_dt = design_dt * first_dt_factor
        n_required_for_reaching_design_dt = int(np.ceil(np.log( 1/first_dt_factor ) / np.log( growth )))
        print('n_required_for_reaching_design_dt =',n_required_for_reaching_design_dt)

        dt_vec = [min(design_dt, first_dt * growth**k) for k in range(n_required_for_reaching_design_dt+1) ]
        t_vec = np.cumsum(dt_vec).tolist()
        t_vec.insert(0,0.)

        time_remaining = tf - t_vec[-1]
        while time_remaining > design_dt:
            t_vec.append( t_vec[-1] + design_dt )
            time_remaining = tf - t_vec[-1]
        if time_remaining > 0:
            t_vec.append( t_vec[-1] + time_remaining )

        t_vec = np.array(t_vec)
        dt_vec = np.diff( t_vec )
        plt.figure()
        plt.semilogy(t_vec[:-1], np.diff(t_vec), marker='.')
        plt.grid()
        plt.xlabel('t (s)')
        plt.ylabel('dt (s)')
        plt.xlim(0,tf)

    # if growth==1:
    #     t_vec = np.linspace(0, tf, nt+1)
    # else:
    #     # import pdb; pdb.set_trace()
    #     dt0 = (1/growth) * tf * (1-growth)/(1-growth**nt)
    #     dt = dt0*(growth**np.array(range(nt+1)))
    #     t_vec = np.cumsum(dt)-dt0
    #     assert t_vec.size == nt+1
    #     assert np.allclose(t_vec[-1], tf, rtol=1e-13)

    # dt = np.diff(x)
    dt = t_vec[1] - t_vec[0] # first dt

    coupler = SpringMassCoupler(kspring=kspring,mass=mass,L0=L0, d=d)
    conductor = Orchestrator(coupler=coupler, order=order)
    
    if bExplicitCoupling: # explicit coupling
        conductor.interfaceSolver = ExplicitSolver
        conductor.NITER_MAX = 1
    else: # implicit coupling
        conductor.interfaceSolver = AitkenUnderrelaxationSolver
        #conductor.interfaceSolver = DampedNewtonSolver
        conductor.NITER_MAX = 1000
        conductor.waveform_tolerance = 1e-11

    for p in conductor.preds:
       p.reset()

    print(mode)
    if mode=="preinit": # initialise predictors with past reference solution
        reset_predictors=False
        t_sample = [ jj*dt for jj in range(-(order+1),1,1) ]
        y_sample = [ conductor._getCouplingVars(t=tt, y=sol_fully_coupled.sol(tt)) for tt in t_sample]
        # print(' t=', t_sample)

        for current_t, current_y in zip(t_sample, y_sample):
          assert current_t > sol_fully_coupled.t[0], 'better not extrapolate the reference solution --> decrease dt or increase tf_transient'
          # add this point to predictors' data
          conductor._advancePredictors(t=current_t,
                                       coupling_vars=current_y)

        init_preds = [p.clone() for p in conductor.preds]
        reset_predictors=False

    elif mode=="postinit": # initialise predictors with future reference solution
        reset_predictors=False
        for p in conductor.preds:
           p.reset()
        t_sample = []
        y_sample = []
        for jj in range(0,order): # just to be sure we have added enough points
          # get interpolated solution
          t_sample.append( jj*dt )
          assert t_sample[-1] < sol_fully_coupled.t[-1], 'better not extrapolate the reference solution --> decrease dt or increase tf_transient'
          y_sample.append( sol_fully_coupled.sol(t_sample[-1]) )


        t_sample = np.array(t_sample)
        y_sample = np.array(y_sample)
        # print(' t=', t_sample)
        # print(' uf=', conductor._getCouplingVars(t=t_sample[-1], y=y_sample[-1]))

        # polynomial fit to create coherent negative-time values
        coef = np.polyfit(x=t_sample, y=y_sample, deg=order-1)
        tvals = np.sort(-t_sample) # in negative time
        yvals = np.array([np.polyval(coef[:,i],tvals) for i in range(y0_global.size)]).T

        # plt.figure()
        # plt.plot(t_sample, y_sample)
        # plt.grid()

        for current_t, current_y in zip(tvals, yvals):
            # add this point to predictors' data
            conductor._advancePredictors(t=current_t,
                                         coupling_vars=conductor._getCouplingVars(t=current_t, y=current_y) )
        init_preds = [p.clone() for p in conductor.preds]

    elif mode=="postinit_iterative": # initialise predictors with future coupled solution (needs iteration)
        reset_predictors=False

        i_iter=0
        error_norm = np.inf
        tol = conductor.waveform_tolerance
        old_end_values = np.array([0.,0.])
        while error_norm > 1:
            i_iter += 1

            # recover previous solution
            t_sample = [ii*dt for ii in range(order)]
            if i_iter==1:
                y_sample = [conductor._getCouplingVars(t=tt, y=y0_global) for tt in t_sample]
                sol = None
            else:
                y_sample = [conductor._getCouplingVars(t=sol.t[it], y=sol.y[:,it]) for it in range(order)]

            t_sample = np.array(t_sample)
            y_sample = np.array(y_sample)
            # old_end_values = y_sample[-1,:]

            assert(len(t_sample)==order)

            # polynomial fit to create coherent negative-time values
            coef = np.polyfit(x=t_sample, y=y_sample, deg=order-1)
            tvals = np.sort(-t_sample) # in negative time
            yvals = np.array([np.polyval(coef[:,i],tvals) for i in range(y_sample[0].size)]).T
            # Feed them to the conductor as initial sampling points
            for p in conductor.preds:
               p.reset()
            for current_t, current_couplingvars in zip(tvals, yvals):
                # add this point to predictors' data
                conductor._advancePredictors(t=current_t,
                                             coupling_vars=current_couplingvars)

            # Perform the simulation for a few steps
            sol = conductor.basic_integration(y0=y0_global.copy(), t_vec=t_sample, #t_vec[:order],
                                  reset_predictors=False)
            # import pdb; pdb.set_trace()
            assert np.allclose(t_sample[-1], sol.t[-1], rtol=1e-13), 'end times do not match...'

            new_end_values = np.array([p.evaluate(sol.t[-1])[0] for p in conductor.preds])
            error_norm = np.linalg.norm( (new_end_values-old_end_values) / (tol + tol*abs(old_end_values)) / old_end_values.size)
            # print(f'  it {i_iter}: error = {error_norm:.2e}')
            # print('   new_end_values=',new_end_values)
            old_end_values = new_end_values

            if i_iter>100:
                raise Exception('Initialisation did not converge')
            # print(f'  it {i_iter}: error = {error_norm:.2e}')
            # print( '    zk  =',y_sample)
            # print( '    zkp1=',np.array([conductor._getCouplingVars(t=sol.t[it], y=sol.y[:,it]) for it in range(order)]))
        print(f'  it {i_iter}: error = {error_norm:.2e}')

        # print(' t=', t_sample)
        # print(' uf=', y_sample[-1,:])
        # reset to negative time
        for p in conductor.preds:
            p.reset()
        for current_t, current_couplingvars in zip(tvals, yvals):
            # add this point to predictors' data
            conductor._advancePredictors(t=current_t,
                                         coupling_vars=current_couplingvars)

        init_preds = [p.clone() for p in conductor.preds]
        
    elif "postinit_iterative_new_nm" in mode: # version pérenne, comprendre n "moins" X
        temp = int(mode.replace('postinit_iterative_new_nm',''))
        nt = order - temp
        print(f'   nt={nt}')
        reset_predictors=False
        conductor.iterative_high_order_initialisation(y0=y0_global.copy(), t0=t_vec[0], dt=dt, nt=nt)
        init_preds = [p.clone() for p in conductor.preds]
        
    elif mode=="noinit":
        reset_predictors=True
        # get initial predictors (mimic the first integration step)
        conductor.basic_integration(y0=y0_global.copy(), t_vec=t_vec[:1], reset_predictors=True)
        init_preds = [p.clone() for p in conductor.preds]

    else:
        raise Exception(f'issue, mode {mode} unknown')

    # print('preds.x = ')
    # print([p.x for p in init_preds])
    # print('preds.y = ')
    # print([p.y for p in init_preds])
    # print(' u(dt) = ', [p.evaluate(dt)[0] for p in init_preds])
    # Coupled simulation
    sol = conductor.basic_integration(y0=y0_global.copy(), t_vec=t_vec,
                                      reset_predictors=reset_predictors)
    assert sol.success, 'simulation did not succeed'
    return sol, init_preds

if __name__=='__main__':

    #%%
    bExplicitCoupling = True
    # sol_postinit_iter,  pred_postinit_iter  = performCoupledSimulation(mode='postinit_iterative',  bExplicitCoupling=bExplicitCoupling)
    sol_postinit_iter, pred_postinit_iter = performCoupledSimulation(mode='postinit_iterative_new_nm2', bExplicitCoupling=bExplicitCoupling)
    sol_boundary, pred_boundary = performCoupledSimulation(mode='noinit', first_dt_factor=1e-3, growth=2., bExplicitCoupling=bExplicitCoupling)
    sol_noinit,   pred_noinit   = performCoupledSimulation(mode='noinit', bExplicitCoupling=bExplicitCoupling)
    sol_preinit,  pred_preinit  = performCoupledSimulation(mode='preinit', bExplicitCoupling=bExplicitCoupling)
    sol_postinit, pred_postinit = performCoupledSimulation(mode='postinit', bExplicitCoupling=bExplicitCoupling)



    #%% Graphical comparison
    plt.figure()
    ivar = 2
    plt.plot(sol_fully_coupled.t, sol_fully_coupled.y[ivar,:], color='tab:orange', linestyle='--', label=r'monolithic')
    if 1:
        plt.plot(sol_preinit.t,  sol_preinit.y[ivar,:],  color='tab:blue',  linestyle='-', marker='o', linewidth=2, label=r'pre init')
        plt.plot(sol_postinit.t, sol_postinit.y[ivar,:], color='tab:green', linestyle='-', marker='o', linewidth=2, label=r'post init')
        plt.plot(sol_postinit_iter.t, sol_postinit_iter.y[ivar,:], color='tab:purple', linestyle='-', marker='o', linewidth=2, label=r'post init iter')
        plt.plot(sol_noinit.t,   sol_noinit.y[ivar,:],   color='tab:red',   linestyle='-', marker='o', linewidth=2, label=r'no init')
        plt.plot(sol_boundary.t,   sol_boundary.y[ivar,:],   color=[0,0,0],   linestyle='-', marker='o', linewidth=2, label=r'boundary')
    if ivar%2 == 0: # for positions only
        ylims = plt.ylim()
        t_eval = np.linspace(-tf/3, tf, 1000)
        plt.plot(t_eval,  pred_preinit[ivar//2].evaluate(t_eval)[0,:],  color='tab:blue',  linestyle='--', linewidth=2, label=r'pred pre init')
        plt.plot(t_eval,  pred_postinit[ivar//2].evaluate(t_eval)[0,:], color='tab:green', linestyle='--', linewidth=2, label=r'pred post init')
        plt.plot(t_eval,  pred_postinit_iter[ivar//2].evaluate(t_eval)[0,:], color='tab:purple', linestyle=':', linewidth=4, label=r'pred post init iter')
        plt.plot(t_eval,  pred_noinit[ivar//2].evaluate(t_eval)[0,:],   color='tab:red',   linestyle='--', linewidth=2, label=r'pred no init')
        plt.plot(t_eval,  pred_boundary[ivar//2].evaluate(t_eval)[0,:],   color=[0,0,0],   linestyle='--', linewidth=2, label=r'pred boundary')
        plt.ylim(ylims[0],ylims[1])
        plt.ylim()
    else:
        plt.autoscale(enable=True, axis='y', tight=True)
    plt.xlim(-tf/2,tf)

    plt.legend(ncol=2, framealpha=0.4, loc='lower left')
    plt.grid()
    # plt.ylim(0,1)
    plt.xlabel('t (s)')
    plt.ylabel('position (m)')
    plt.title('Dynamics of the dual spring-mass system')
