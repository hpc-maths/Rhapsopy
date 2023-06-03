#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:21:09 2023

@author: lfrancoi
"""
from rhapsopy.coupling import FIXEDPOINT
from rhapsopy.accelerators.base import BaseFixedPointSolver

class ExplicitSolver(BaseFixedPointSolver):
  def __init__(self, logger, *args):
    super().__init__(logger)

  def solve(self, fun, x0, **kwargs):
    self.logger.log(FIXEDPOINT,'Explicit solver: performing single iteration')
    return fun(x0), 1
