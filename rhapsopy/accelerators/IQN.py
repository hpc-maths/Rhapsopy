#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:18:32 2023

@author: lfrancoi
"""
import numpy as np
from rhapsopy.coupling import FIXEDPOINT, WRnonConvergence
from rhapsopy.accelerators.base import BaseFixedPointSolver
import scipy

class IQNSolver(BaseFixedPointSolver):
  def __init__(self, logger):
    super().__init__(logger)
    raise Exception('Not implemented')