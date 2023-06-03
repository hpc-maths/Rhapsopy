#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:37:31 2021

Configuration of a simple heat conduction test case.
Two 1D solid slabs exchange heat through a (potentially reactive) surface
The properties are assumed constant in each slab.

   solid 1      interface    solid 2
================== || ------------------

@author: laurent
"""
import numpy as np
from rhapsopy.examples.heat1D.model.heat import setupFiniteVolumeMesh

def getSetup(N1=20, N2=20, rho1_sur_rho2=50., rho2=1., bReactive=True, init_profile=1):
    #%% Solid n°1
    options1={}
    options1['solid'] = {
              'mesh':{},
               'BCs':{
                 'right':{
                         'type': None,
                         'Timposed': None,
                         'flux': None,
                         },
                 'left':{
                         'type': 'neumann',
                         'Timposed': lambda t:0.2,
                         'flux': lambda t: 0.,
                        },
              },
              'rho': rho1_sur_rho2 * rho2,
              'cp': 1.,
              'lbda': 1.,
             }
    options1['solid']['D'] = options1['solid']['lbda']/(options1['solid']['rho']*options1['solid']['cp'])

    options1['coupling'] = {'right':{}, 'left':{}}

    #%% Solid n°2
    options2={}
    options2['solid'] = {
              'mesh':{},
               'BCs':{
                 'right':{
                         'type': 'neumann',
                         'Timposed': lambda t: 0.2,
                         'flux': lambda t: 0.,
                         },
                 'left':{
                         'type': None,
                         'Timposed': None,
                         'flux': None,
                        },
              },
              'rho': rho2,
              'cp': 1.,
              'lbda': 1.,
             }
    options2['solid']['D'] = options2['solid']['lbda']/(options2['solid']['rho']*options2['solid']['cp'])

    options2['coupling'] = {'right':{}, 'left':{}}

    #%% Volume source terms
    options1['solid']['source'] = lambda t,x: 0*x
    options2['solid']['source'] = lambda t,x: 0*x

    # Mesh, Initial State
    if not bReactive: # no reactive surface

        xfaces=np.linspace(-1,0,N1)
        options1['solid']['mesh'] = setupFiniteVolumeMesh(xfaces=xfaces)

        xfaces=np.linspace(0,1,N2)
        options2['solid']['mesh'] = setupFiniteVolumeMesh(xfaces=xfaces)

        # initial state
        if init_profile==1:
            options1['y0']= 1*np.exp(-options1['solid']['mesh']['cellX']**2)
            options2['y0']= 1*np.exp(-options2['solid']['mesh']['cellX']**2)
            phi_surf = lambda t,Ts: 0.
        else:
            options1['y0']= 0.5+0.5*np.tanh(options1['solid']['mesh']['cellX']*5)
            options2['y0']= 0.5+0.5*np.tanh(options2['solid']['mesh']['cellX']*2)
            phi_surf = lambda t,Ts: 0.

    else:
        #%% Meshes and initial solution
        ## Non-uniform mesh
        # xfaces=np.logspace(-3,0,N1)
        # xfaces= xfaces-xfaces[0]
        # xfaces = -xfaces[::-1]
        # xfaces=np.logspace(-5,0,N2)
        # xfaces= xfaces-xfaces[0]
        ## Uniform mesh
        xfaces1=np.linspace(-1,0,N1)
        xfaces2=np.linspace(0,1,N2)

        options1['solid']['mesh'] = setupFiniteVolumeMesh(xfaces=xfaces1)
        options2['solid']['mesh'] = setupFiniteVolumeMesh(xfaces=xfaces2)

        #%% Surface heat release term
        #T0 = 0.2; Tac = 1.; A = 1e2
        # T0 = 0.24; Tac = 3.; A = 2e3
        T0 = 0.35; Tac = 3.; A = 2e3
        # T0 = 0.21; Tac = 2.; A = 1e3
        # T0 = 0.2; Tac = 1.5; A = 1e2
        Tb = 1.
        phi_surf = lambda t,Ts,Tac=Tac,Tb=Tb: A * np.exp(-Tac/Ts) * (Ts-Tb)

        #%% initial state
        options1['y0']= T0 + 0.*options1['solid']['mesh']['cellX']
        options2['y0']= T0 + 0.*options2['solid']['mesh']['cellX']

        #%% Source terms
        options1['solid']['source'] = lambda t,x: 0*x
        options2['solid']['source'] = lambda t,x: 0*x

        
    #%% Useful data for interface coupling
    options1['coupling']['right']['a1'] = options1['solid']['lbda'] / ( options1['solid']['mesh']['faceX'][-1]  - options1['solid']['mesh']['cellX'][-1] )
    options1['coupling']['right']['a2']  = options2['solid']['lbda'] / ( options2['solid']['mesh']['faceX'][0]   - options2['solid']['mesh']['cellX'][0]  )
    options1['coupling']['right']['phi_surf']  = phi_surf

    options2['coupling']['left']['a1'] = options1['coupling']['right']['a2']
    options2['coupling']['left']['a2']  = options1['coupling']['right']['a1']
    options2['coupling']['left']['phi_surf']  = lambda t,Ts: -phi_surf(t,Ts) # flip it !!!!

    return options1, options2

if __name__=='__main__':
  import matplotlib.pyplot as plt
  options1,options2 = getSetup(init_profile=0, bReactive=True)

  plt.figure()
  plt.plot(options1['solid']['mesh']['cellX'], options1['y0'], marker='+')
  plt.plot(options2['solid']['mesh']['cellX'], options2['y0'], marker='+')
  plt.grid()
  plt.xlabel('x')
  plt.ylabel('T')
  plt.title('Initial solution for the 1D heat transfer')
  
  #%% Monolithic solution
  from coupler_heat import Coupler
  import scipy.integrate
  y0_global = np.hstack( (options1['y0'], options2['y0']) )
  x_global =  np.hstack( (options1['solid']['mesh']['cellX']-options1['solid']['mesh']['cellX'][-1],
                          options2['solid']['mesh']['cellX']) )
  nx1 = options1['y0'].size
  nx2 = options2['y0'].size

  coupler = Coupler(options1, options2, coupling_modes=('dirichlet', 'dirichlet'))
  t_span = [0,1e4]

  outRef = scipy.integrate.solve_ivp(fun=coupler.coupledODE,
                                    y0=y0_global, method='LSODA', #t_eval=np.linspace(t_span[0], t_span[1], 10),
                                    t_span=t_span, atol=1e-8, rtol=1e-8, uband=2, lband=2, dense_output=True,
                                    vectorized=False, args=(), jac=None)
  sol, time = outRef.y, outRef.t
  print('solution obtained in {} fev'.format(outRef.nfev))
  print('  --> "{}"'.format(outRef.message))
  if not outRef.success:
      raise Exception('Reference integration failed')

  outRef = coupler.complementOutput(outRef)
  # y1, y2 = sol[:nx1,:], sol[nx1:,:]
  # E1 = getEnergy(y1,options1)
  # E2 = getEnergy(y2,options2)
  # Etot = E1 +E2
  # Ts = getTs(coupling=options1['coupling']['right'],
  #         T1=y1[-1,:],
  #         T2=y2[0,:],
  #         t=time)
  

  plt.figure()
  plt.semilogx(outRef.t, outRef.Ts)
  plt.xlabel('t (s)')
  plt.ylabel('Ts')
  plt.grid()