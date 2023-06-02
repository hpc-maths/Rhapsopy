#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:21:22 2023

Acceleration methods for the fixed-point problem

@author: lfrancoi
"""
from .IQN import IQNSolver
from .aitken_dynamicrelaxation import AitkenUnderrelaxationSolver
from .explicit import ExplicitSolver
from .fixed_point import FixedPointSolver, AitkenScalarSolver
from .newton import NewtonSolver
from .damped_newton import DampedNewtonSolver
from .anderson import AndersonSolver
from .base import BaseFixedPointSolver
from .JFNK import JFNKSolver
