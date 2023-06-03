#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:54:51 2023

@author: lfrancoi
"""

import numpy as np
from rhapsopy.coupling import FIXEDPOINT, WRnonConvergence, ExceptionWhichMayDisappearWhenLoweringDeltaT
from rhapsopy.accelerators.base import BaseFixedPointSolver
import scipy.optimize
from scipy.optimize.nonlin import NoConvergence

class JFNKSolver(BaseFixedPointSolver):
  def __init__(self, logger):
    super().__init__(logger)

  def solve(self, fun, x0, ftol, rtol, maxiter, args=()):
    global ncalls
    ncalls = 0
    def resfun(x): # residuals
      global ncalls
      ncalls += 1
      if ncalls>maxiter:
        raise NoConvergence()
      return x-fun(x)

    try:
      inner_M = None # no preconditionner
      sol = scipy.optimize.newton_krylov(F=resfun, xin=x0, iter=None, rdiff=max(1e-6,rtol/10),
                               method='lgmres', inner_maxiter=maxiter+1, inner_M=inner_M, outer_k=100,
                               verbose=False, maxiter=maxiter, f_tol=1e-12,
                               f_rtol=None, x_tol=rtol, x_rtol=rtol, tol_norm=None,
                               line_search=None,
                               callback=None)
    except NoConvergence as e:
      print(e)
      msg = "JFNK Failed to converge after {} calls".format(ncalls)
      self.logger.critical(msg)
      raise WRnonConvergence(msg)
      
    except (OverflowError, ValueError) as e:
      print(e)
      msg = "JFNK internal error after {} calls ({})".format(ncalls,e)
      self.logger.critical(msg)
      raise ExceptionWhichMayDisappearWhenLoweringDeltaT(msg)
      
    return sol, ncalls