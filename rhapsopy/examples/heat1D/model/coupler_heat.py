#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 10:10:44 2022

Coupler object for the case of ehat conduction between two 1D solid slabs

@author: lfrancoi
"""
import rhapsopy.coupling
import numpy as np
from  rhapsopy.examples.heat1D.model.heat import heatModelfunODE, getTs
import scipy.integrate
from scipy.interpolate import lagrange

import logging
from scipy.integrate._ivp.ivp import METHODS

INTEGRATION_DETAIL = 20
INTEGRATION_DETAIL2 = 15
INTEGRATION = 30

class FakePredictor():
  def __init__(self, val):
    self.val = val

  def evaluate(self, t):
    return self.val

class Coupler(rhapsopy.coupling.BaseCoupler):
    """ This class is the interface between the generic code-coupling-simulation
    class and the subsystem solvers.
    It is physics-aware and connects two heat conduction solvers. """
    def __init__(self, options1, options2, coupling_modes, solverclass='LSODA'):
        # Create logger object
        self.logger = logging.getLogger('rhapsopy.heat.coupler')
        self.logger.handlers = [] # drop previous handlers
        self.logger.setLevel(100) # no logging by default

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(1) # no filtering
        formatter = logging.Formatter('%(name)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.log(INTEGRATION_DETAIL, 'Initialising coupler object')


        # option structures for the subsytems
        self.options = [options1, options2] # option structure for both subsystems
        self.options1 = options1
        self.options2 = options2
        self.nSubsystems = 2

        #############################
        # Find which coupling variables are required based on interface conditions
        allPossibleVars = {'Ts': False,
                           'phim': False, 'phip': False,
                           'Tm': False, 'Tp': False,
                            'Ts_extrap_left': False, 'Ts_extrap_right': False,
                            'phi_extrap_left': False, 'phi_extrap_right': False,
                           } # which predictions are required

        self.extrapolateRight = False
        self.extrapolateLeft  = False
        ### left slab
        if coupling_modes[0]=='dirichlet': # need Ts
            allPossibleVars['Ts'] = True
        if coupling_modes[0]=='neumann': # need heat flux entering left slab
            allPossibleVars['phim'] = True # not phip !!!
        if coupling_modes[0]=='balance': # need to predict T in first right cell
            allPossibleVars['Tp'] = True

        # additional modes which work for the non-reactive case mostly (needs more work to work with reactive)
        if coupling_modes[0]=='dirichlet_extrap': # need Ts extrapolated from right thermal profile
            allPossibleVars['Ts_extrap_right'] = True
            self.extrapolateRight = True
        if coupling_modes[0]=='neumann_extrap': # need flux extrapolated from right thermal profile
            allPossibleVars['phi_extrap_right'] = True
            self.extrapolateRight = True

        ### right slab
        if coupling_modes[1]=='dirichlet': # need Ts
            allPossibleVars['Ts'] = True
        if coupling_modes[1]=='neumann': # need heat flux entering right slab
            allPossibleVars['phip'] = True # not phim !!!
        if coupling_modes[1]=='balance': #need to predict T in last left cell
            allPossibleVars['Tm'] = True

        if coupling_modes[1]=='dirichlet_extrap': # need Ts extrapolated from left thermal profile
            allPossibleVars['Ts_extrap_left'] = True
            self.extrapolateLeft = True
        if coupling_modes[1]=='neumann_extrap': # need flux extrapolated from left thermal profile
            allPossibleVars['phi_extrap_left'] = True
            self.extrapolateLeft = True

        self.couplingVarDict  = dict([(key, {"value": None, "required": required}) for key, required in allPossibleVars.items()])
        self.couplingVarNames = [key for key in allPossibleVars if allPossibleVars[key]]
        self.nCouplingVars = len( self.couplingVarNames )
        #############################


        self.nx = [opts['solid']['mesh']['cellX'].size for opts in self.options]
        self.coupling_modes = coupling_modes
        self.predictors = None # Dictionnary form of the polynomial predictors used by the code coupling class, for better handling

        # Choice of integration for each subproblem
        self.adaptive_subsolves = True # if True, adaptative time stepping is used for each solver
        if isinstance(solverclass,str):
          solverclass = METHODS[solverclass]
        self.subsystem_solvers = [None for i in range(self.nSubsystems)]
        self.method = solverclass # integration method for each subsystem

        self.nsubsolves = [0 for i in range(self.nSubsystems)]

    def getTs(self,t,y):
      """ Computes surface temperature """
      nx1 = self.options1['solid']['mesh']['cellX'].size
      y1 = y[:nx1]
      y2 = y[nx1:]
      Ts1 = getTs(self.options1['coupling']['right'],
          T2=y2[0],
          T1=y1[nx1-1],
          t=t)
      Ts2 = getTs(self.options2['coupling']['left'],
          T1=y2[0],
          T2=y1[nx1-1],
          t=t)
      assert abs(Ts2-Ts1)/(1e-10 + 1e-10*abs(Ts1))<1, f'Ts are not coherent: Ts1={Ts1} while Ts2={Ts2}'
      return Ts1

    def getAdditionalVars(self,t,y):
        """ Compute surface temperature and heat flux """
        nx1 = self.options1['solid']['mesh']['cellX'].size
        y1 = y[:nx1]
        y2 = y[nx1:]
        Ts1 = getTs(self.options1['coupling']['right'],
            T2=y2[0],
            T1=y1[nx1-1],
            t=t)
        Ts2 = getTs(self.options2['coupling']['left'],
            T1=y2[0],
            T2=y1[nx1-1],
            t=t)
        assert abs(Ts2-Ts1)/(1e-10 + 1e-10*abs(Ts1))<1, f'Ts are not coherent: Ts1={Ts1} while Ts2={Ts2}'

        Ts = Ts1
        phis = self.options1['coupling']['right']['phi_surf'](t,Ts)
        Tm = y1[-1]
        Tp = y2[0]
        phim = (Ts-Tm) * self.options1['coupling']['right']['a1']
        phip = (Ts-Tp) * self.options2['coupling']['left']['a1']

        # Find Ts and phi based on spatial extrapolations
        extrap_stencil = 2
        if self.extrapolateLeft:
            xL = self.options1['solid']['mesh']['cellX']
            poly_L = lagrange( xL[-extrap_stencil:], y1[-extrap_stencil:] )
            Ts_extrap_left = poly_L(0.)
            phi_extrap_left = self.options1['solid']['lbda'] * np.imag(poly_L(0.+1e-50*1j)) * 1e50
            phis_extrap_left = self.options1['coupling']['right']['phi_surf'](t,Ts_extrap_left)
            phi_extrap_left += phis_extrap_left
        else:
            Ts_extrap_left = None
            phi_extrap_left = None
            phis_extrap_left = None

        if self.extrapolateRight:
            xR = self.options2['solid']['mesh']['cellX']
            poly_R = lagrange( xR[:extrap_stencil], y2[:extrap_stencil] )
            Ts_extrap_right = poly_R(0.)
            phi_extrap_right = self.options2['solid']['lbda'] * np.imag(poly_R(0.+1e-50*1j)) * 1e50
            phis_extrap_right = self.options1['coupling']['right']['phi_surf'](t,Ts_extrap_right)
            phi_extrap_right -= phis_extrap_right
        else:
            Ts_extrap_right = None
            phi_extrap_right = None
            phis_extrap_right = None

        self.couplingVarDict['Ts']['value'] = Ts
        self.couplingVarDict['Tm']['value'] = Tm
        self.couplingVarDict['Tp']['value'] = Tp
        self.couplingVarDict['phim']['value'] = phim
        self.couplingVarDict['phip']['value'] = phip

        self.couplingVarDict['Ts_extrap_left']['value']  = Ts_extrap_left
        self.couplingVarDict['phi_extrap_left']['value']  = phi_extrap_left
        self.couplingVarDict['Ts_extrap_right']['value'] = Ts_extrap_right
        self.couplingVarDict['phi_extrap_right']['value']  = phi_extrap_right

        return [k['value'] for key, k in self.couplingVarDict.items() if k['required']]

    def coupledODE(self,t,y,imex_mode=-1):
        """ Model function for the monolithic approach, using the predictor interface """
        pred_vals = self.getAdditionalVars(t,y)
        preds = [ FakePredictor(val) for val in  pred_vals ]

        dy1dt = self.integrateSingleSubsystem(isolv=0, t0=t, y0=y, dt=None, preds=preds, rtol=None, bDebug=False, bReturnDydt=True, imex_mode=imex_mode)
        dy2dt = self.integrateSingleSubsystem(isolv=1, t0=t, y0=y, dt=None, preds=preds, rtol=None, bDebug=False, bReturnDydt=True, imex_mode=imex_mode)

        # construct overall time derivatives
        dydt = np.hstack((dy1dt, dy2dt))
        return dydt

    def f1(self,t,y):
        """ Model function for IMEX methods, f1 returns the time derivatives of each subsytem, without the coupling terms """
        return self.coupledODE(t,y,imex_mode=0)

    def f2(self,t,y):
        """ Model function for IMEX methods, f1 returns the time derivatives of each subsytem, without the coupling terms """
        return self.coupledODE(t,y,imex_mode=1)


    def _spreadPredictors(self, pred_list):
        """ Take the generic list of predictors from the code coupling class, and organise it into a physics-aware dictionnary """
        self.logger.log(INTEGRATION_DETAIL, 'spreading predictors')
        self.predictors = {'left':{}, 'right':{}}
        for i,name in enumerate(self.couplingVarNames):
            if name=='Ts':
                self.predictors['Ts'] = pred_list[i]

            elif name=='phim':
                self.predictors['left']['flux'] = pred_list[i]
            elif name=='phip':
                self.predictors['right']['flux'] = pred_list[i]

            elif name=='Tm':
                self.predictors['left']['Tbnd'] = pred_list[i]
            elif name=='Tp':
                self.predictors['right']['Tbnd'] = pred_list[i]

            elif name=='Ts_extrap_left':
                self.predictors['left']['Ts_extrap'] = pred_list[i]
            elif name=='Ts_extrap_right':
                self.predictors['right']['Ts_extrap'] = pred_list[i]

            elif name=='phi_extrap_left':
                self.predictors['left']['phi_extrap'] = pred_list[i]
            elif name=='phi_extrap_right':
                self.predictors['right']['phi_extrap'] = pred_list[i]

            else:
              raise Exception(f'coupling var {i} "{name}" unknown')

    def _feedOptions(self):
        """ Handle the boundary conditions for each solver """
        self.logger.log(INTEGRATION_DETAIL, 'feeding options')
        for i in range(2): # loop on subsystems
            if i==0:
                coupling_side = 'right'; side_of_neighbor = 'left'; predictor_side='left';
                index_bnd=-1; index_bnd2=0;
                opts1=self.options1; opts2=self.options2; #current_y=y1; other_y=y2;
            else:
                coupling_side = 'left';  side_of_neighbor = 'right'; predictor_side='right';
                index_bnd=0;  index_bnd2=-1;
                opts1=self.options2; opts2=self.options1; #current_y=y2; other_y=y1;


            ## fill in the coupling data
            # safety first
            opts1['coupling'][coupling_side]['T2'] = None
            opts1['solid']['BCs'][coupling_side]['Timposed'] = None
            opts1['solid']['BCs'][coupling_side]['flux'] = None

            opts1['solid']['BCs'][coupling_side]['type'] = self.coupling_modes[i].split('_')[0] # get rid of potential extrapolation


            if self.coupling_modes[i]=='balance': # balance coupling
              opts1['solid']['BCs'][coupling_side]['type']='balance'
              opts1['coupling'][coupling_side]['T2'] = lambda t, pred=self.predictors[coupling_side]['Tbnd']:  pred.evaluate(t)
                #= lambda t, pred=pred:  pred.evaluate(t)
            elif self.coupling_modes[i]=='dirichlet': # Dirichlet coupling
              opts1['solid']['BCs'][coupling_side]['type']='dirichlet'
              opts1['solid']['BCs'][coupling_side]['Timposed'] = lambda t, pred=self.predictors['Ts']:  pred.evaluate(t)

            elif self.coupling_modes[i]=='neumann': # Neumann coupling
              opts1['solid']['BCs'][coupling_side]['type']='neumann'
              opts1['solid']['BCs'][coupling_side]['flux'] = \
                  lambda t, pred=self.predictors[predictor_side]['flux']: pred.evaluate(t)

            elif self.coupling_modes[i]=='neumann_extrap':
              opts1['solid']['BCs'][coupling_side]['type']='neumann'
              opts1['solid']['BCs'][coupling_side]['flux'] = \
                        lambda t, pred=self.predictors[coupling_side]['phi_extrap']: pred.evaluate(t)

            elif self.coupling_modes[i]=='dirichlet_extrap':
              opts1['solid']['BCs'][coupling_side]['type']='dirichlet'
              opts1['solid']['BCs'][coupling_side]['Timposed'] = \
                  lambda t, pred=self.predictors[coupling_side]['Ts_extrap']: pred.evaluate(t)

            else:
                raise Exception('coupling mode "{}" not found'.format(self.coupling_modes[i]))

        self.options1['coupling']['right']['phi_cor'] = lambda t: 0.
        self.options1['coupling']['left']['phi_cor']  = lambda t: 0.
        self.options2['coupling']['right']['phi_cor'] = lambda t: 0.
        self.options2['coupling']['left']['phi_cor']  = lambda t: 0.

    def getCouplingVars(self,t,y):
        """ Call each system's object to gather the required coupling variables """
        self.logger.log(INTEGRATION_DETAIL, 'getting coupling vars')
        return np.array( self.getAdditionalVars(t=t, y=y) )

    def partialStateUpdate(self, isolv, y, yk):
        """ Update part of the state vector (with a new state vector for the isolv-th subsystem) """
        nx1 = self.options1['y0'].size
        yup = y.copy()
        if isolv==0:
          yup[:nx1] = yk
        else:
          yup[nx1:] = yk
        return yup

    def complementOutput(self, out):
        """ Add useful data to simulation output """
        from rhapsopy.examples.heat1D.model.heat import getTs, getEnergy
        nx1 = self.options1['y0'].size
        out.y1, out.y2 = out.y[:nx1,:], out.y[nx1:,:]
        out.E1 = getEnergy(out.y1,self.options1)
        out.E2 = getEnergy(out.y2,self.options2)
        out.Etot = out.E1 + out.E2
        # Ts = out.z[0,:]
        out.Ts = getTs(coupling=self.options1['coupling']['right'],
                    T2=out.y2[0,:],
                    T1=out.y1[-1,:],
                    t=out.t)
        return out

    def integrateSingleSubsystem(self, isolv, t0, y0, dt, preds, rtol=None, bDebug=False,
                                 bReturnDydt=False, imex_mode=-1):
        """ Performs one iteration of a code coupling step (Jacobi or Gauss-Seidel)
              --> computes the value of the overall state vector at time t+dt,
                  starting from state y at time t. """
        self.logger.log(INTEGRATION_DETAIL, f'coupler: integrating subsystem {isolv}')
        self._spreadPredictors(preds)
        ## fill in the coupling data
        self._feedOptions()
        self.nsubsolves[isolv] += 1

        ## get each subsystem's state vector
        nx1 = self.options1['y0'].size
        y1 = y0[:nx1]
        y2 = y0[nx1:]
        y0_list=[y1,y2]

        fun = lambda t,x: heatModelfunODE(t=t, x=x, options=self.options[isolv],
                                          imex_mode=imex_mode, sideOfCoupling=['right', 'left'][isolv])
        y0  = y0_list[isolv]

        if bReturnDydt: # return overall state vector time derivative
          # useful for testing, and to use a monolithic approach as reference
          return fun(t0,y0)

        # else, actually integrate the subsytem forward in time
        if rtol is None: rtol=1e-13
        atol = rtol

        # Integrate each system separately
        bandhalf = 1 # half width of the diagonal jacobian

        # Define solvers
        if self.adaptive_subsolves: #ADAPTIVE INTEGRATION
            integrator = lambda fun,y0 : scipy.integrate.solve_ivp( fun=fun, y0=y0,
                                              t_span=[t0,t0+dt], max_step=dt, first_step=dt/2, #1e-10,
                                              method=self.method, atol=atol, rtol=rtol, uband=bandhalf, lband=bandhalf)
        else: # single step integration
          if 0: # explicit
              integrator = lambda fun,y0 : scipy.integrate.solve_ivp( fun=fun, y0=y0,
                                                t_span=[t0,t0+dt], max_step=dt, first_step=dt,
                                                method='RK23', atol=1e50, rtol=1e50, uband=bandhalf, lband=bandhalf)
          else: # implicit
              integrator = lambda fun,y0 : scipy.integrate.solve_ivp( fun=fun, y0=y0,
                                                t_span=[t0,t0+dt], max_step=dt, first_step=dt,
                                                method='Radau', atol=1e0, rtol=1e0, uband=bandhalf, lband=bandhalf)

        if bDebug: # debug first dy/dt
            dydt0 = fun(t0,y0)
            print('dydt0[{}]='.format(isolv),dydt0)
            return None

        try:
          current_out = integrator(fun=fun, y0=y0)
        except ValueError as e:
          if 'first_step' in str(e):
            import pdb; pdb.set_trace()
          raise e

        if not current_out.success:
          raise rhapsopy.coupling.ExceptionWhichMayDisappearWhenLoweringDeltaT(current_out.message)

        if not self.adaptive_subsolves:
          assert current_out.t.size<3, 'single step integration has made multiple steps...'

        self.logger.log(INTEGRATION_DETAIL, 'integration completed')

        return current_out
