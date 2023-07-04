#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:06:36 2023

Dual mass-spring system test case

@author: lfrancoi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from rhapsopy.coupling import Orchestrator, BaseCoupler


def odefun_single(t,x,kspring,mass,L0,d,inputs):
    # x = [position, speed]
    dxdt = np.zeros((2,))
    dxdt[0] = x[1]
    dxdt[1] = (1/mass) * ( -d*x[1] + kspring*(inputs['xip1'](t)-x[0]-L0) - kspring*(x[0]-inputs['xim1'](t)-L0) )
    return dxdt

class SpringMassCoupler(BaseCoupler):
    def __init__(self,kspring, mass, L0, d=0.):
        self.kspring = kspring
        self.mass = mass
        self.d = d
        self.L0 = L0
        self.nCouplingVars = 2 # we have 2 coupling variables: the positions of both masses
        self.nSubsystems   = 2 # we have 2 coupled subsystems

        # fully coupled system
        self.A = np.array(((0.,1,0,0),
            (-2*kspring/mass,-d/mass,kspring/mass,0),
            (0,0,0,1),
            (kspring/mass,0,-2*kspring/mass,-d/mass)))
        self.b = np.array((0.,0,0,self.kspring/self.mass))

    def coupled_ode(self,t,y):
        """ Fully coupled system """
        return self.A @ y + self.b

    def getCouplingVars(self,t,y):
        """ Return the coupling variables, which are simply the speed and positions of the nodes """
        return y[::2]

    def partialStateUpdate(self, isolv, y, yk):
        """ Update part of the state vector (with a new state vector for the isolv-th subsystem) """
        yup = y.copy()
        if isolv==0:
          yup[:2] = yk
        else:
          yup[2:] = yk
        return yup
      
    def integrateSingleSubsystem(self, isolv, t0, y0, dt, preds, rtol=None, bDebug=False):
        """ Performs one iteration of a code coupling step (Jacobi or Gauss-Seidel)
              --> computes the value of the overall state vector at time t+dt,
                  starting from state y at time t. """
        # get each subsystem's state vector
        y0_subsystems = [y0[2*i:2*(i+1)] for i in range(2)]

        # Distribute the predictions
        current_input = {}
        if isolv==0: # first mass
            current_input['xim1'] = lambda t: 0. # left attachment point
            current_input['xip1'] = lambda t: preds[1].evaluate(t)
        else: # second mass
            current_input['xim1'] = lambda t: preds[0].evaluate(t)
            current_input['xip1'] = lambda t: 1. # right attachment point

        # perform time integration
        #if rtol is None: rtol=self.rtol
        rtol = atol = 1e-12
        current_out = solve_ivp(fun=lambda t,y : odefun_single(t, y,
                                                               kspring=self.kspring,
                                                               mass=self.mass,
                                                               L0=self.L0,
                                                               d=self.d,
                                                               inputs=current_input),
                                y0=y0_subsystems[isolv], t_span=[t0,t0+dt],
                                max_step=dt/2, first_step=dt/2, #TODO: sometimes first_step=dt raises issues in Scipy
                                method='DOP853', atol=atol, rtol=rtol)
        if not current_out.success:
            raise Exception(current_out.message)
        return current_out

    def coupled_ode2(self,t,y):
        """ Coupeld ODE using the subsystem ODEs """
        # get each subsystem's state vector
        y_subsystems = [y[2*i:2*(i+1)] for i in range(2)]

        # Distribute the predictions
        dydt=np.zeros_like(y)
        for isolv in range(2):
            current_input = {}
            if isolv==0: # first mass
                current_input['xim1'] = lambda t: 0. # left attachment point
                current_input['xip1'] = lambda t: y_subsystems[1][0]
            else: # second mass
                current_input['xim1'] = lambda t: y_subsystems[0][0]
                current_input['xip1'] = lambda t: 1. # right attachment point

            dydt[2*isolv:2*(isolv+1)] = odefun_single(t=t, x=y_subsystems[isolv],
                                                   kspring=self.kspring,
                                                   mass=self.mass, L0=self.L0,
                                                   d=self.d, inputs=current_input)
        return dydt


if __name__=='__main__':
    # Simple comparison of coupled and monolithic solutions
    kspring=10.
    mass=1.
    L0=1.
    d=0.33

    coupler = SpringMassCoupler(kspring=kspring,mass=mass,L0=L0,d=d)
    y0_global = np.array([0.3, 0, 0.8, 0])
    tf = 10.

    from rhapsopy.accelerators import ExplicitSolver
    conductor = Orchestrator(coupler=coupler, order=3)
    conductor.interfaceSolver = ExplicitSolver
    conductor.NITER_MAX = 1

    # compute analytical solution
    sol = solve_ivp(fun=coupler.coupled_ode,
                    y0=y0_global, t_span=[0.,tf],
                    method='DOP853', atol=1e-12, rtol=1e-12)

    sol2 = solve_ivp(fun=coupler.coupled_ode2,
                y0=y0_global, t_span=[0.,tf],
                method='DOP853', atol=1e-12, rtol=1e-12)

    assert np.allclose( coupler.coupled_ode(sol.t[-1], sol.y[:,-1]), coupler.coupled_ode2(sol.t[-1], sol.y[:,-1]) ), 'issue'

    # Compute coupled solution
    nt = 200 # number of coupling time steps
    solcoupled = conductor.basic_integration(y0=y0_global, t_vec=np.linspace(0,tf,nt))

    # Graphical comparison
    plt.figure()
    plt.plot(sol.t, sol.y[0,:], color='tab:orange', label=r'$y_0$ monolithic')
    plt.plot(sol.t, sol.y[2,:], color='tab:blue', label=r'$y_1$ monolithic')
    plt.plot(sol2.t, sol2.y[2,:], color='tab:green', linewidth=4, label=r'$y_1$ monolithic2')
    plt.plot(solcoupled.t, solcoupled.y[0,:], color='tab:orange', linestyle='--', linewidth=3, label=r'$y_0$ coupling')
    plt.plot(solcoupled.t, solcoupled.y[2,:], color='tab:blue', linestyle='--', linewidth=3, label=r'$y_1$ coupling')
    plt.ylim(0,1)
    plt.legend(ncol=2)
    plt.grid(); plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('t (s)')
    plt.ylabel('position (m)')
    plt.title('Dynamics of the dual spring-mass system')
