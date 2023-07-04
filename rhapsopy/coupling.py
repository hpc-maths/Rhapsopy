#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:03:05 2021

Test two slabs coupled at an interface with code coupling techniques or with a monolithic approach

@author: laurent
"""
import numpy as np
np.seterr(divide="raise")
np.set_printoptions(precision=13)
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.optimize import OptimizeResult as OdeResult
import time as pytime
import traceback, sys

import logging
logging.raiseExceptions = True

from rhapsopy.prediction import predicteur

# logging levels
FIXEDPOINT = 11
ORCHESTRATOR = 31
INTEGRATION = 31 # start and end of integration
INTEGRATION_DETAIL = 21 # integration steps
INTEGRATION_DETAIL2 = 16 # more info on convergence
INTEGRATION_DETAIL3 = 11 # more info on convergence
INTEGRATION_DETAIL4 = 6 # detailed information about everything
INTEGRATION_DETAIL5 = 3 # even more everything :)


SINGLE_STEP_FORWARD = 1
COMPLETE_STEP_FORWARD = 3
COMPLETE_STEP_FORWARD2 = 2
COUPLING_VAR_UPDATES = 1
PREDICTOR_UPDATES = 1
# TODO: different loggers or filters for each part (fixed-point, integration)

# ERROR CODES
WRNONCONVERGENCE  = 1 # main method failed
WRNONCONVERGENCE2 = 2 # embedded method failed
OTHEREXCEPTION = 5    # other exception during step
ERRORTOOHIGH = 6      # error estimate too large
ACCEPTED = 0          # step is accepted

# Cuisine interne
TOLERANCE_FACTOR_DTOPT = 1.01
MAXRELSTEP = 2.0 # maximum increase factor for the time step
MINRELSTEP = 0.1 # minimum ...
SAFETY_FACTOR = 0.8

ncalls=None
others=None


class WRnonConvergence(Exception):
    """ Exception class used in cased of convergence failure for implicit code-coupling """
    pass

class ExceptionWhichMayDisappearWhenLoweringDeltaT(Exception):
    """ Exception class used for handling other issues (e.g. failures within subsystems) """
    pass

class BaseCoupler():
    """ Model for the coupler class that handles the subsystem integration """
    def __init__(self, options1, options2, coupling_modes):
        self.nCouplingVars = 5 # number of coupling variables
        self.predictors = None # Dictionnary form of the polynomial predictors used by the code coupling class, for better handling

    def getCouplingVars(self,t,y):
        raise NotImplementedError()

    def integrateSingleSubsystem(self, isolv, t0, y0, dt, preds, rtol=None, bDebug=False):
        raise NotImplementedError()

#%% Dense-output interpolant for coupling variables
from rhapsopy.prediction import newton_interp_coef, newton_interp_coef2
from scipy.integrate._ivp import OdeSolution
class Interpolant():
  def __init__(self, tn, tnp1, preds):
    self.tn   = tn
    self.tnp1 = tnp1
    self.nvars = len(preds)
    self.coefs = [p.coef.T for p in preds]
    self.xis = [p.x for p in preds]
    test = [xi.size==coef.size for xi,coef in zip(self.xis,self.coefs)]
    # TODO: adapt for predictors with more than 1 var ?
    if not np.all(test):
      # import pdb; pdb.set_trace()
      raise Exception('issue')

    # test cal DEBUG
    self.__call__( 0.5*(tn+tnp1) )
    self.__call__( tn + np.linspace(0,1,10)*(tnp1-tn) )


  def __call__(self, t):
    # assert np.all(t>=self.tn)
    # assert np.all(t<=self.tnp1)
    interpedvals = np.vstack( [newton_interp_coef(self.xis[i], t, self.coefs[i]) for i in range(self.nvars)] )
    return interpedvals

#%% Code-coupling wrapper
class Orchestrator:
  """ This class sets up and performs the coupled integration of the multiple subsystems,
  using a code coupling approach to enable high-order time accuracy. """

  def __init__(self, coupler, order, error_method=0):
    """ Initialize the coupled integration """
    self._init_log()

    assert order > 0, 'prescribed coupling order of accuracy must be > 0'
    self.order = order
    self.logger.log(ORCHESTRATOR, f'maximum prediction order set to {self.order}')

    self.coupler = coupler
    self.preds = [ predicteur(NMAX=order) for i in range(coupler.nCouplingVars) ]
    self.error_method = error_method
    for p in self.preds:
      p.error_method = self.error_method

    # WR iteration
    # parameters
    self.waveform_tolerance = 1e-8 # tolerance for the termination fo the waveform iterations
    self.minimum_waveform_tolerance = np.inf # minimum tolerance for the waveform relaxation
    self.gauss_seidel = False # WR iteration scheme
    self.subsystem_ordering = list(range(coupler.nSubsystems)) # solver ordering for WR iteration scheme (only meaningful if Gauss-Seidel mode is activated)
    self.raise_error_on_non_convergence = True # raise an error if the WR iteration do not converge
    self.NITER_MAX = 1000 # maximum number of WR iterations per step
    self.enforce_last_call = False # if True, one additionnal solver call is executed with the converged coupling variables to ensure everything is coherent (may improve Newton)
    self.subsystemsIntegratedInSingleCall = False
    # If True, the function "integrateAllSubsystems" of the coupler object
    # is called to integrate all subsystems, as for a parallel Jacobi-type simulation.
    # Otherwise, the function "integrateSingleSubsystem" is called to integrate
    # the subsystems one by one.

    # convergence acceleration
    self.bInitSolver = True # to activate solver initialisation at first run
    self.interfaceSolver = None
    self.checkConvergenceRate = True # if True, WR convergence rate is monitored

    # adaptive integration
    self.embedded_method = False # if True, the coupling error is estimated by comparing the coupling variables
    self.maximise_order_increase = False # if True, for WR coupling, the first iteration is directly second-order accurate (the initial point is not dropped)
    self.higher_order_embedded = True # obtained for different approximation orders (better for implicit coupling)
    self.deadzone = [1., 1.] # time step deadzone, i.e. minimal time step ratio for a time step change to be performed

  def _init_log(self):
    # create logger
    self.logger = logging.getLogger("rhapsopy.coupling")
    self.logger.handlers = [] # drop previous handlers
    self.logger.setLevel(100) # no logging by default

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(1) # no filtering of logs
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(formatter)
    self.logger.addHandler(ch)
    self.logger.log(ORCHESTRATOR, 'Set up coupling logger')

    # idem for solver
    self.solverlogger = logging.getLogger("rhapsopy.coupling.solver") # inherits all of the above setup
    self.solverlogging = lambda msg: self.solverlogger.log(ORCHESTRATOR, msg)
    self.solverlogger.log(ORCHESTRATOR, 'Set up solver logger')

  def _init_solver(self):
    """ Initialise fixed-point solver """
    # TODO: the user should already instantiate the solver, so as to be able to precisely control its internal parameters
    self.logger.log(INTEGRATION_DETAIL3, f'Initialising solver of class {self.interfaceSolver}')
    from rhapsopy.accelerators import BaseFixedPointSolver
    assert issubclass(self.interfaceSolver, BaseFixedPointSolver)
    self.solvers = [self.interfaceSolver(self.solverlogger) for i in range(4)] # we instantiate several solvers, as may be required for the embedded error estimate

  def _step_forward(self, t, yn, dt, values_last_iterate,state_last_iterate=None, rtol=None):
    """ Performs one iteration of a code coupling step (Jacobi or Gauss-Seidel)
          --> computes the value of the overall state vector at time t+dt,
              starting from state y at time t. """
    self.logger.log(SINGLE_STEP_FORWARD, 'performing single step iteration')
    # Update the value of the last point (remember that we are now working in interpolation mode based on the previous
    # iteration of the current time step)
    self._updatePredictors(t=t+dt, coupling_vars=values_last_iterate)

    if self.subsystemsIntegratedInSingleCall: # all subsystems are integrated in a Jacobi manner at once
        self.logger.log(SINGLE_STEP_FORWARD, ' subsystems will be integrated in a single coupler call')
        outs = self.coupler.integrateAllSubsystems(t0=t, y0=yn, dt=dt, preds=self.preds, rtol=rtol)
        newy = [o.y[:,-1] for o in outs]

    else: # Jacobi or Gauss-Seidel approach, one subsystem at a time
        outs = [None for i in range(len(self.subsystem_ordering))]
        newy = [None for i in range(len(self.subsystem_ordering))]
        for it, isolv in enumerate(self.subsystem_ordering):
            self.logger.log(SINGLE_STEP_FORWARD, f' integrating subsystem {isolv}')
            outs[isolv] = self.coupler.integrateSingleSubsystem(isolv=isolv, t0=t, y0=yn, dt=dt, preds=self.preds, rtol=rtol)
            newy[isolv] = outs[isolv].y[:,-1]

            if self.gauss_seidel and ( it < len(self.subsystem_ordering)-1 ): # do not update after the last solve
              if state_last_iterate is None:
                ynp1_intermediaire = self.coupler.partialStateUpdate(isolv=isolv, y=yn, yk=newy[isolv])
              else: # use the previous value (last fixed-point function call)
                ynp1_intermediaire = self.coupler.partialStateUpdate(isolv=isolv, y=state_last_iterate, yk=newy[isolv])
              self._updatePredictors(t=t+dt, coupling_vars=self._getCouplingVars(t=t+dt, y=ynp1_intermediaire))
              # raise Exception('Current implementation of the Gauss-Seidel scheme messes up with the convergence and the prediction')
              # If the coupling variables which are used by the next subsystem are dependent on this next system,
              # then this approach introduces an O(dt) error, since we use evaluate the coupling variables with
              # its state at time tn, not tn+dt...

              # TODO: the updated values should be computed from the new state (left or right) we just computed,
              # and from the prediction of the other one !

              # TODO: an alternative could be to use the state vector at time tn+dt from the previous iteration if available

              # TODO: be able to only compute coupling vars prescribed by a single solver instead of calling all subsystems

    self.logger.log(SINGLE_STEP_FORWARD, '     subsystem substeps: {}'.format([o.t.size-1 for o in outs]))
    self.logger.log(SINGLE_STEP_FORWARD, 'step iteration performed')

    # construct overall state vector
    return np.hstack(newy)

  def perform_step(self, y0, t, dt, atol_iter=None, rtol_iter=None, bDebug=False, rtol=None, embedded=False, solver_number=0):
        """ Perform the WR iteration process for a single time step """
        if self.bInitSolver: # initialise fixed-point solver
          self._init_solver()
          self.bInitSolver = False

        self.logger.log(COMPLETE_STEP_FORWARD, 'performing step')
        tn = t # current starting time
        tnp1 = t+dt # next coupling time

        ynp1_kp1 = y0 # TODO: better guess

        if dt<1e-12:
            raise Exception('stop dt is too low')
        assert not (self.preds[0].x is None), 'predictors have not been initialised !!!'


        if atol_iter is None:
          atol_iter = self.waveform_tolerance
        if rtol_iter is None:
          rtol_iter = self.waveform_tolerance

        pred_coupling_vars = np.array([p.evaluate(tnp1, allow_outside=True) for p in self.preds])[:,0]

        solver = self.solvers[solver_number]
        ## Since we are at the first iteration, we only have data up to time t
        # and extrapolate on [t, t+dt]
        # To do so:
        # - we first register the solution at time t once at the first iteration
        #   of the new time step (which starts from t and goes to t+dt)
        # - then we evaluate the extrapolation at t+dt and register a fake point
        #   at t+dt with these values (this drops the oldest interpolation point,
        #   but does not affect the prediction polynomial)
        # - then, for the other iterations of the current time step, we simply
        #   update these last data points with the current iteration's values,
        #   so that we work in interpolation mode instead !
        # 1 - We append these points
        if embedded:
          # we do nothing here, since the embedded approach already provides the proper perdictor setup
          pass
        else:
          if abs((self.preds[0].x[0] - tnp1)/dt) < 1e-2:
                raise Exception('Time tnp1 seems to have already been registered in the predictors...')
          else:
            oldNs = [p.N for p in self.preds]
            self._advancePredictors(t=tnp1, coupling_vars=pred_coupling_vars)
            # discard the oldest point so that the polynomial is unaffected (in particular its order...)
            # however, we must set N back to its correct value after the WR iterations are done and the step is accepted
            if self.maximise_order_increase:
              pass
            else: # drop the oldest point
              for oldN,pred in zip(oldNs,self.preds):
                  pred.setCurrentN(oldN)

        global ncalls
        global others
        ncalls=0
        others={}
        others["ynp1_kp1"]=None
        def fixedPointFunction(coupling_vars_k, full_output=False):
            """ Fixed point function.
                Inputs:
                    - coupling_vars_k:  array
                        guess of the coupling variables at time t_{n+1}
                    - full_output: boolean
                        if True, additional outputs are given
                Output:
                    - coupling_vars_kp1: array
                        the updated coupling variables at t_{n+1}, obtained
                        after integrating the subsystems separately from time
                        t_{n} to t_{n+1} and recomputing the coupling variables
                        based on the new final states
                    - ynp1_kp1: array
                        new overall state vector
            """
            # TODO: it would be more sensible for large-scale applications to
            # let each solver object store (if asked by the orchestrator) its data
            # and only transmit to the coupler the  coupling variables.
            # The coupler would ask each solver to update, store, and validate
            # (after step acceptance) its state vector.

            global ncalls
            global others
            ncalls+=1
            ynp1_kp1 = self._step_forward(yn=y0, t=tn, dt=dt,
                                          values_last_iterate=coupling_vars_k,
                                          state_last_iterate=others["ynp1_kp1"],
                                          rtol=rtol)
            coupling_vars_kp1 = self._getCouplingVars(t=tnp1, y=ynp1_kp1)
            others["ynp1_kp1"] = ynp1_kp1
            if full_output:
                return coupling_vars_kp1, ynp1_kp1
            else:
                return coupling_vars_kp1

        # Accelerated fixed-point
        coupling_vars_kp1 = np.array([p.evaluate(tnp1)[0] for p in self.preds])
        coupling_vars_kp1, n_iter = solver.solve(fun=fixedPointFunction,
                                             x0=coupling_vars_kp1.copy(),
                                             ftol=1e-15, rtol=rtol_iter,
                                             maxiter=self.NITER_MAX)
        ynp1_kp1 = others["ynp1_kp1"]
        # coupling_vars_kp1 = np.array([p.evaluate(tnp1)[0] for p in self.preds])

        # just for debug purposes (check that this does not change the Newton output)
        if self.enforce_last_call:
            # ensure that y is computed for the last coupling variables
            # --> ensures the coherence of the implicit results
            # --> for explicit coupling, this is equivalent to performing a second iteration...
            ___, ynp1_kp1 = fixedPointFunction(coupling_vars_kp1, full_output=True)

        coupling_vars_kp1_pred = np.array([p.evaluate(tnp1)[0] for p in self.preds])
        self.logger.log(COMPLETE_STEP_FORWARD2, 'error between last call and converged coupling vars: {:.2e}'.format(
          np.max( np.abs( (coupling_vars_kp1-coupling_vars_kp1_pred)/(1e-10 + abs(coupling_vars_kp1_pred)) ) )))

        self.logger.log(COMPLETE_STEP_FORWARD2, 'coupling_vars_kp1 at convergence = '+str(coupling_vars_kp1))
        self.logger.log(COMPLETE_STEP_FORWARD, f'step performed ({ncalls} subsystem calls)')
        return coupling_vars_kp1, ncalls, ynp1_kp1
        #TODO: beware, ynp1_kp1 is evaluated at coupling_vars_k, not kp1, unless enforce_last_call is True

  def _getCouplingVars(self,t,y):
    """ Call each system's object to gather the required coupling variables """
    self.logger.log(COUPLING_VAR_UPDATES, 'getting coupling variables')
    return self.coupler.getCouplingVars(t,y)

  def _getPredOrders(self):
      """ Return current prediction orders """
      return np.array([p.N for p in self.preds])

  def _updatePredictors(self, t, coupling_vars):
    """ Update the predictors by refreshing the coupling variables at the last sampling point """
    self.logger.log(PREDICTOR_UPDATES, 'updating predictors')
    for p, var in zip(self.preds, coupling_vars):
        assert np.allclose(p.x[0], t)
        p.replaceLastPoint(x=t, y=var)

  def _advancePredictors(self, t, coupling_vars, bIncreaseN=True):
    """ Add a new sampling point to the predictors """
    self.logger.log(INTEGRATION_DETAIL4, 'advancing predictors with new coupling variables')
    for p, var in zip(self.preds, coupling_vars):
        p.appendData(x=t, y=var, bIncreaseN=bIncreaseN)

  def _backupPredictors(self):
    """ Backup the current predictors, so as to be able to restore them in case of step failure,
    avoiding the possibility of conveying errors due to sampling point modifications """
    self.logger.log(INTEGRATION_DETAIL4, 'backing up predictors')
    return [p.clone() for p in self.preds]

  def _restorePredictors(self, preds):
    """ Restore the predictors to a previous backup """
    self.logger.log(INTEGRATION_DETAIL4, 'restoring predictors')
    self.preds = [p.clone() for p in preds]


  def basic_integration(self, y0, t_vec, reset_predictors=True, nDebugAfterNsteps=np.inf,
                        high_order_iter_init=False,
                        bDenseOutput=False):
    """ Basic integration with prescribed time steps (t_vec) with waveform relaxation for each step """
    self.logger.log(INTEGRATION, 'Starting coupled simulation with prescribed time steps')
    tstart = pytime.time()
    nt = t_vec.size
    interpolants = []

    iter_hist = [] # WR iterations per step
    t0 = t_vec[0]

    if reset_predictors:
      # reset predictors to ensure they do not contain previously acquired data
      # else, assume they have been properly setup
      for p in self.preds:
        p.reset()
      # assert self.preds[0].x is None, 'Predictors seem to have been initialised already'
      # add the starting point as initial data for the predictors
      ini_coupling_vars = self._getCouplingVars(t=t0, y=y0)
      self._advancePredictors(t=t0, coupling_vars=ini_coupling_vars)

    if high_order_iter_init:
      presol = self.iterative_high_order_initialisation(y0=y0, t0=t0, dt=t_vec[1]-t_vec[0])
      yhist = [presol.y[:,i] for i in range(presol.t.size)]
      couplingvar_hist = [presol.z[:,i] for i in range(presol.t.size)]
      istart = len(yhist)
      if bDenseOutput:
        for i in range(0,istart-1):
          interpolants.append( Interpolant(t_vec[i], t_vec[i+1], self.preds) )

      for nite in presol.WR_iters:
        iter_hist.append(nite)
    else:
      istart = 1
      yhist = [y0]
      couplingvar_hist = [ np.hstack([p.evaluate(t0) for p in self.preds]) ]


    ##### Temporal loop #####
    bFailed = False
    for i in range(istart, nt): # go from tn to tn+1
      bDebug = nDebugAfterNsteps<i # if True, make lots of plot and quit
      tn   = t_vec[i-1]
      tnp1 = t_vec[i]
      dt   = tnp1 -  tn
      self.logger.log(INTEGRATION, f"t={tn:.3e} s, dt={dt:.2e} s (step {i}/{nt-1})")

      backup_preds = self._backupPredictors()

      try:
          coupling_vars_np1, niter, ysol = self.perform_step(y0=yhist[i-1], t=tn, dt=dt, bDebug=bDebug)
      except ExceptionWhichMayDisappearWhenLoweringDeltaT as e:
        self.logger.log(INTEGRATION_DETAIL, f'Unexpected exception: {e}')
        if self.raise_error_on_non_convergence:
          raise e
        else:
          bFailed = True
          break

      except WRnonConvergence as e:
        self.logger.log(INTEGRATION_DETAIL, 'Loop has not converged')
        if self.raise_error_on_non_convergence:
          raise e
        else:
          bFailed = True
          break

      yhist.append( np.copy(ysol) )
      iter_hist.append(niter)
      couplingvar_hist.append( coupling_vars_np1 )
      if bDenseOutput:
          interpolants.append( Interpolant(tn, tnp1, self.preds) )


      self.logger.log(INTEGRATION_DETAIL, f'  converged in {niter} iterations')
      # Update predictors
      if 0: # assume everything is well handled :)
          self._updatePredictors(t=tnp1, coupling_vars=coupling_vars_np1)
          # In this case, the new predictors already have a "fake" interpolation point at tnp1,
          # for which we decreased the number N of sampling points so that the order is unchanged.
          # We now recover the "oldest" data point for the predictors, so that we may
          # increase the order of the prediction !
          for pred in self.preds:
              pred.increaseN()
      else: # safer: restore predictor backups
          self._restorePredictors(backup_preds)
          self._advancePredictors(t=tnp1, coupling_vars=coupling_vars_np1)

    tend = pytime.time()
    out = OdeResult()
    out.success = not bFailed
    if not bFailed:
        out.message = 'Coupled simulation successfully reached end time'
        out.t = t_vec
    else:
        out.message = 'Early exit due to non convergence'
        out.t = t_vec[:i]

    self.logger.log(INTEGRATION, out.message)
    out.y = np.array(yhist).T
    out.z = np.array(couplingvar_hist).T
    if bDenseOutput:
      out.solz = OdeSolution(out.t, interpolants)
    out.WR_iters = np.array( iter_hist )
    out.nWRtotal = np.sum( out.WR_iters )
    out.CPUtime = tend-tstart
    return out


  def _compute_new_solution(self, y0, z_sample, t_sample, t_sample_neg):
      # polynomial fit to create coherent negative-time values
      coef = np.polyfit(x=t_sample, y=z_sample, deg=t_sample.size-1)
      tvals = t_sample_neg # in negative time
      zvals = np.array([np.polyval(coef[:,i],tvals) for i in range(z_sample[0].size)]).T
      # Feed them to the conductor as initial sampling points
      for p in self.preds:
         p.reset()
      for current_t, current_couplingvars in zip(tvals, zvals):
          # add this point to predictors' data
          self._advancePredictors(t=current_t,
                                  coupling_vars=current_couplingvars)

      # Perform the simulation for a few steps
      sol = self.basic_integration(y0=y0.copy(), t_vec=t_sample,
                            reset_predictors=False, high_order_iter_init=False)
      assert np.allclose(t_sample[-1], sol.t[-1], rtol=1e-13), 'end times do not match...'

      z_sample_new = sol.z.T

      return z_sample_new, zvals, sol
      # return sol

  def iterative_high_order_initialisation(self, y0, t0, dt, nt=None):
        """ Iterative high-order initialisation: the first few steps are run multiple times until
        the high-order interpolant on these steps converges """
        if nt is None:
          nt = self.order
        i_iter=0
        order = self.order
        error_norm = np.inf
        tol = self.waveform_tolerance
        old_end_values = None
        t_sample = np.array( [t0 + ii*dt for ii in range(nt)] )
        t_sample_neg = np.sort(np.array( [t0 - ii*dt for ii in range(nt)] )) # in negative time

        # prepare first iteration
        a = self._getCouplingVars(t=t0, y=y0)
        z_sample = np.array([a for tt in t_sample])
        old_end_values = a
        WR_iters = [0 for i in range(nt-1)]
        assert len(t_sample)==nt

        # TODO: enable the execution all the time steps at once, it may spare CPU cost ?
        while error_norm > 1:
            i_iter += 1
            if i_iter>100:
                raise Exception('Initialisation did not converge')

            # recover previous solution
            z_sample_kp1, zvals, sol = self._compute_new_solution(y0, z_sample, t_sample, t_sample_neg)
            for i in range(nt-1):
              WR_iters[i] += sol.WR_iters[i]

            new_end_values = np.array([p.evaluate(t_sample[-1])[0] for p in self.preds])
            assert np.allclose(new_end_values, z_sample_kp1.T[:,-1])
            error_norm = np.linalg.norm( (new_end_values-old_end_values) / (tol + tol*abs(old_end_values)) / old_end_values.size)
            print(f'  it {i_iter}: error = {error_norm:.2e}')
            sys.stdout.flush()
            old_end_values = new_end_values
            z_sample = z_sample_kp1

        # reset to negative time
        for p in self.preds:
            p.reset()
        for current_t, current_couplingvars in zip(t_sample_neg, zvals):
            # add this point to predictors' data
            self._advancePredictors(t=current_t,
                                    coupling_vars=current_couplingvars)
        sol.WR_iters = WR_iters
        return sol

  def adaptive_integration(self, y0, t_span, atol, rtol, reset_predictors=True,
                           first_step=1e-6, max_step=np.inf,
                           max_nt=np.inf,
                           nPrintLevel=np.inf, bDebug=False, keepUniformSampling=False,
                           bDenseOutput=False,
                           high_order_iter_init=False):
    """ Adaptive integration with prescribed error tolerances """

    self.logger.log(INTEGRATION, "Adaptive integration begins")

    assert len(t_span)==2
    tstart = pytime.time()
    t0 = t_span[0]

    if reset_predictors:
      for p in self.preds:
        p.reset()
      # add the starting point as initial data for the predictors
      self._advancePredictors(t=t0, coupling_vars=self._getCouplingVars(t=t0, y=y0))
    for p in self.preds:
      p.setTolerances(atol=atol,rtol=rtol)

    ##### Temporal loop #####
    yhist  = [y0]
    thist = [t_span[0]]
    interpolants = []
    interpolants_init, interpolants_embd1, interpolants_embd2 = [], [], [] # for the embedded method
    interpolants_true_embd1, interpolants_true_embd2 = [], []
    couplingvar_hist = [ np.hstack([p.evaluate(t0) for p in self.preds]) ]
    iter_hist = [] # WR iterations per step
    iter_hist2 = [] # WR iterations per step for the embedded method
    p_hist = [] #[p.N for p in self.preds]] # prediction orders

    tn = t_span[0]
    dt = first_step
    i=0
    nsteps_total = 0
    nsteps_rejected = 0
    nsteps_accepted = 0
    nsteps_failed = 0
    step_info = []


    ratio_rtol_toliter = 0.1
    ratio_rtol_rtolsubsys = ratio_rtol_toliter * 0.2
    ratio_toliter_rtolsubsys = ratio_rtol_toliter / ratio_rtol_rtolsubsys
    # ratio_rtol_toliter = 1./5
    # ratio_rtol_rtolsubsys = 1./20


    while tn<t_span[-1]:
      bAccepted = False
      ntry=1
      i+=1
      if i>max_nt:
        break
      consecutive_fails = -1
      self.logger.log(INTEGRATION_DETAIL,        f"\ttn={tn:.3e} s (step {i})")
      initial_extraps = self._backupPredictors()# to keep the original extrapolated values

      while not bAccepted:
        self.logger.log(INTEGRATION_DETAIL2,f"\t tn={tn:.3e} s, dt={dt:.3e} (try {ntry})")
        if keepUniformSampling: # create fictious sampling points that remain equidistant
            for p in initial_extraps:
                new_times = [tn + ii*dt for ii in range(-p.Nacquired+1,1)]
                new_vals = [p.evaluate(tt, allow_outside=True) for tt in new_times]
                oldN = p.N
                oldNacquired = p.Nacquired
                p.reset()
                for tt,yy in zip(new_times, new_vals):
                    p.appendData(x=tt, y=yy)
                p.setCurrentN(oldN)
                assert oldNacquired == p.Nacquired

        tnp1 = tn + dt # next coupling time
        self._restorePredictors(preds=initial_extraps)
        # if consecutive_fails>2: # lower the order of the predictors
        #   self.logger.log(INTEGRATION_DETAIL2,'Lowering the prediction order due to repeated failures')
        #   for p in self.preds:
        #     p.reportDiscontinuity()
        #   initial_extraps = self._backupPredictors()

        try:
          # TODO: each solver should store yn and ynp1, and have the ability to revert to yn
          # TODO: this makes it necessary to make sure that the last call to the subsolvers is the right one,
          # and that the corresponding coupling vars are used afterwards !
          coupling_vars_np1, niter, ysol = self.perform_step(y0=yhist[i-1], t=tn,
                                                      dt=dt, bDebug=bDebug,
                                                      rtol = min(self.minimum_waveform_tolerance * ratio_toliter_rtolsubsys,
                                                                 rtol * ratio_rtol_rtolsubsys),
                                                      atol_iter=min(self.minimum_waveform_tolerance,
                                                                    rtol * ratio_rtol_toliter),
                                                      rtol_iter=min(self.minimum_waveform_tolerance,
                                                                    rtol * ratio_rtol_toliter),
                                                      solver_number=0)
        except ExceptionWhichMayDisappearWhenLoweringDeltaT as e:
            self.logger.log(INTEGRATION_DETAIL2, ' issue during WR iteration --> lowering time step')
            self.logger.log(INTEGRATION_DETAIL3, f' error was {e}')
            self.logger.log(INTEGRATION_DETAIL3, f' Traceback was {traceback.format_exc()}')
            step_info.append((tn,dt,self._getPredOrders(),OTHEREXCEPTION))

            dt=dt/4
            nsteps_failed+=1
            nsteps_total+=1
            consecutive_fails+=1
            continue

        except WRnonConvergence:
            self.logger.log(INTEGRATION_DETAIL3, 'WR Loop has not converged --> lowering time step')
            step_info.append((tn,dt,self._getPredOrders(),WRNONCONVERGENCE))

            dt=dt/4
            nsteps_failed+=1
            nsteps_total+=1
            consecutive_fails+=1
            continue

        except Exception as e:
            # other exceptions, which have not been handled by the coupled object
            self.logger.critical('unexpected issue during WR iteration (embedded method) --> lowering time step')
            self.logger.critical(f' error was {e}')
            self.logger.critical(f' Traceback was {traceback.format_exc()}')
            raise e

        truly_used_preds = self._backupPredictors()

        #TODO: update must be done for explicit coupling if no embedded error estimate !!!
        self._updatePredictors(t=tnp1, coupling_vars=coupling_vars_np1)
        main_preds = self._backupPredictors()

        if self.embedded_method:
            # perform the step anew with a higher-order approximation to use as error estimate
            self.logger.log(INTEGRATION_DETAIL3,"\t computing embedded solution")
            old_orders = self._getPredOrders()
            self._restorePredictors(initial_extraps)
            if self.higher_order_embedded or min([p.N for p in self.preds])<2:
              for p in self.preds:
                  p.changeNmax(p.NMAX+1)
              self._advancePredictors(t=tnp1, coupling_vars=coupling_vars_np1)

            else:
              for p in self.preds:
                  oldN = p.N
                  p.changeNmax(oldN-1)
              self._advancePredictors(t=tnp1, coupling_vars=coupling_vars_np1)
   
            # feed the new point at t_{n+1} to increase the stencil and increment the order
            new_orders = self._getPredOrders()

            if not all(abs(new_orders-old_orders)==1):
              raise Exception('issue with embedded order')
            try:
              coupling_vars_np1_2, niter2, ysol2 = self.perform_step(y0=yhist[i-1], t=tn,
                                                      dt=dt, bDebug=False,
                                                      rtol = min(self.minimum_waveform_tolerance * ratio_toliter_rtolsubsys,
                                                                 rtol * ratio_rtol_rtolsubsys),
                                                      atol_iter=min(self.minimum_waveform_tolerance,
                                                                    rtol * ratio_rtol_toliter),
                                                      rtol_iter=min(self.minimum_waveform_tolerance,
                                                                    rtol * ratio_rtol_toliter),
                                                      solver_number=1,
                                                      embedded=True)

            except ExceptionWhichMayDisappearWhenLoweringDeltaT as e:
                self.logger.log(INTEGRATION_DETAIL2, ' issue during WR iteration (embedded method) --> lowering time step')
                self.logger.log(INTEGRATION_DETAIL3, f' error was {e}')
                self.logger.log(INTEGRATION_DETAIL3, f' Traceback was {traceback.format_exc()}')
                step_info.append((tn,dt,self._getPredOrders(),OTHEREXCEPTION))

                dt=dt/4
                nsteps_failed+=1
                nsteps_total+=1
                consecutive_fails+=1
                continue

            except WRnonConvergence:
                self.logger.log(INTEGRATION_DETAIL3, 'WR Loop has not converged (embedded method) --> lowering time step')
                step_info.append((tn,dt,self._getPredOrders(),WRNONCONVERGENCE2))

                dt=dt/4
                nsteps_failed+=1
                nsteps_total+=1
                consecutive_fails+=1
                continue

            except Exception as e:
                # other exceptions, which have not been handled by the coupled object
                self.logger.critical('unexpected issue during WR iteration (embedded method) --> lowering time step')
                self.logger.critical(f' error was {e}')
                self.logger.critical(f' Traceback was {traceback.format_exc()}')
                raise e

            truly_used_preds_embd = self._backupPredictors()

            self._updatePredictors(t=tnp1, coupling_vars=coupling_vars_np1_2)
            embedded_preds = self._backupPredictors()
            self._restorePredictors(main_preds)

        #### Error control
        # compute error based on the comparison of the extrapolated and
        # converged coupling variables
        dt_opts = []
        dt_opts2 = []
        self.logger.log(INTEGRATION_DETAIL3,"\t estimating coupling errors")
        for ii in range(len(self.preds)):
            if self.embedded_method:
              new_pred = embedded_preds[ii]
              old_pred = main_preds[ii]
            else:
              new_pred = main_preds[ii]
              old_pred = initial_extraps[ii]

            if self.logger.isEnabledFor(INTEGRATION_DETAIL5):
              self.logger.log(INTEGRATION_DETAIL5,f'pred {ii}')
              self.logger.log(INTEGRATION_DETAIL5, 'new_pred: '+str(new_pred))
              self.logger.log(INTEGRATION_DETAIL5, 'old_pred: '+str(old_pred))
              self.logger.log(INTEGRATION_DETAIL5, ' new_pred(tnp1)  ='+str(new_pred.evaluate(tnp1)))
              self.logger.log(INTEGRATION_DETAIL5, ' old_pred(tnp1)  ='+str(old_pred.evaluate(tnp1)))
              self.logger.log(INTEGRATION_DETAIL5, ' init_pred(tnp1) ='+str(initial_extraps[ii].evaluate(tnp1)))

            dt_opts.append( old_pred.eval_optimal_timestep(
                                        other_pred=new_pred,
                                        tn=tn, tnp1=tnp1) )
            dt_opts2.append( new_pred.eval_optimal_timestep(
                                        other_pred=old_pred,
                                        tn=tn, tnp1=tnp1) )

        dt_opts3 = np.maximum(dt_opts,dt_opts2)
        dt_opt = min(dt_opts3)
        self.logger.log(INTEGRATION_DETAIL4, '\tdt_opts  = '+str(dt_opts) )
        self.logger.log(INTEGRATION_DETAIL4, '\tdt_opts2 = '+str(dt_opts2))
        self.logger.log(INTEGRATION_DETAIL4, '\tdt_opts3 = '+str(dt_opts3))
        self.logger.log(INTEGRATION_DETAIL3, f'\tdt_opt = {dt_opt:.3e} s')

        bAccepted = dt < TOLERANCE_FACTOR_DTOPT*dt_opt
        if bAccepted:
          step_info.append((tn,dt,self._getPredOrders(),ACCEPTED,dt_opts,dt_opt))
          for p in self.preds:
            p.nsuccess_with_current_order += 1
          p_hist.append( self._getPredOrders() )
          nsteps_accepted+=1
          self.logger.log(INTEGRATION_DETAIL2, '\t==> accepted step')
          yhist.append( np.copy(ysol) )
          couplingvar_hist.append( np.hstack([p.evaluate(tnp1) for p in self.preds]) )
          thist.append( tnp1 )
          iter_hist.append(niter)
          if self.embedded_method:
              iter_hist2.append(niter2)

          if bDenseOutput:
              interpolants.append( Interpolant(tn, tnp1, self.preds) )
              interpolants_init.append( Interpolant(tn, tnp1, initial_extraps) )
              interpolants_embd1.append( Interpolant(tn, tnp1, main_preds) )
              interpolants_true_embd1.append( Interpolant(tn, tnp1, truly_used_preds) )
              if self.embedded_method:
                interpolants_embd2.append( Interpolant(tn, tnp1, embedded_preds) )
                interpolants_true_embd2.append( Interpolant(tn, tnp1, truly_used_preds_embd) )
          # prepare next step
          self._restorePredictors(initial_extraps)
          self._advancePredictors(t=tnp1, coupling_vars=coupling_vars_np1)
          tn=tnp1

        else:
          self.logger.log(INTEGRATION_DETAIL3,"\t step refused (error too large)")
          step_info.append((tn,dt,self._getPredOrders(),ERRORTOOHIGH,dt_opts,dt_opt))
          nsteps_rejected+=1
          ntry+=1

        nsteps_total+=1
        dt_opt = SAFETY_FACTOR*dt_opt
        dt_original = dt
        assert dt_opt>0
        dt=dt_opt

        # deadzone
        if (dt >  dt_original*self.deadzone[0]) and (dt < dt_original*self.deadzone[1]):
          dt = dt_original
          self.logger.log(INTEGRATION_DETAIL3, 'time step unchanged (deadzone)')

        if dt_opt<MINRELSTEP*dt_original:
          self.logger.log(INTEGRATION_DETAIL3, 'time step limited by maximum reduction factor')
          dt = MINRELSTEP*dt_original

        if dt_opt>MAXRELSTEP*dt_original:
          self.logger.log(INTEGRATION_DETAIL3, 'time step limited by maximum increase factor')
          dt = MAXRELSTEP*dt_original

        if bAccepted:
          dt_end = t_span[1]-tnp1
        else:
          dt_end = t_span[1]-tn
        if dt>dt_end:
          self.logger.log(INTEGRATION_DETAIL3, 'time step limited by final time')
          dt = dt_end

        if dt>max_step:
          self.logger.log(INTEGRATION_DETAIL3, 'time step limited by maximum allowed time step')
          dt = max_step

        if dt<0:
          msg = "time step has become negative"
          self.logger.critical(msg)
          raise Exception(msg)

    if i>max_nt:
      self.logger.log(INTEGRATION, 'maximum number of steps reached')
    else:
      self.logger.log(INTEGRATION, 'Adaptive coupled integration has successfully reached end time')
    tend = pytime.time()

    out = OdeResult()
    out.t = np.array(thist)
    out.y = np.array(yhist).T
    assert out.y.ndim == 2
    out.z = np.array(couplingvar_hist).T
    out.success = True
    out.message = 'Success'
    out.WR_iters = np.array( iter_hist )
    if self.embedded_method:
        out.WR_iters2 = np.array( iter_hist2 )
    if bDenseOutput:
        out.solz = OdeSolution(out.t, interpolants)
        out.solz_init = OdeSolution(out.t, interpolants_init)
        out.solz_embd1 = OdeSolution(out.t, interpolants_embd1)
        out.solz_true_embd1 = OdeSolution(out.t, interpolants_true_embd1)
        if self.embedded_method:
          out.solz_embd2 = OdeSolution(out.t, interpolants_embd2)
          out.solz_true_embd2 = OdeSolution(out.t, interpolants_true_embd2)

    out.nsteps_total = nsteps_total
    out.nsteps_rejected = nsteps_rejected
    out.nsteps_accepted = nsteps_accepted
    out.nsteps_failed = nsteps_failed
    out.CPUtime = tend-tstart
    out.p_hist = np.array(p_hist)
    out.step_info = step_info

    return out

def process_step_info(outCoupled):
    """ Post-process step rejections and other information """
    error_codes = []
    step_info = {}
    temp=(
            ('WRNONCONVERGENCE', 1),
            ('WRNONCONVERGENCE2',2),
            ('OTHEREXCEPTION',   5),
            ('ERRORTOOHIGH',     6),
            ('ACCEPTED',         0),
         )
    error_codes = [a[1] for a in temp]

    for failmode, value in  temp:
        step_info[failmode] = {"error_code": value,
                               "tn": [],
                               "dt": [],
                               "orders": [],
                               "dt_opts": [],
                               "dt_opt": [],
                               }

    for d in outCoupled.step_info:
        error_code = d[3]
        ierr = error_codes.index(error_code)
        code = temp[ierr][0]
        tn = d[0]
        dt = d[1]
        orders = d[2]
        if len(d)>4:
            dt_opts = d[4]
            dt_opt  = d[5]

        assert step_info[code]['error_code']==error_code
        step_info[code]["tn"].append(tn)
        step_info[code]["dt"].append(dt)
        step_info[code]["orders"].append(orders)

        if len(d)>4:
            step_info[code]["dt_opts"].append(dt_opts)
            step_info[code]["dt_opt"].append(dt_opt)

    for key1 in step_info.keys():
        for key2 in step_info[key1].keys():
            if isinstance(step_info[key1][key2], list):
                step_info[key1][key2] = np.array(step_info[key1][key2])
    return step_info
