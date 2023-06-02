# Code-coupling demonstrator for conjugate heat transfer

A Python package to simulate the heat transfer between to solid slabs, using code coupling techniques.
This mimics the coupling of two solvers for the simulation of conjugate transfer between a solid material and a gas flow.

To test this package, run `python setup.py develop` to register it as a development package.

#### TODO:
* Fix Automatic order adaptation (both methods)
* Conservativity considerations
* Do not directly handle the susbsytem's state vectors
* Fix strange time step peaks when use adaptive integration in the heat transfer test case
* Implement Newton for the fixed-point problem
* Implement least-square fits for improved stability ?
* Fix Gausse-Seidel scheme
