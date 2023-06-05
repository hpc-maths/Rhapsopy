# Rhapsopy
___
### **R**eally **H**igh-order **A**da**P**tive coupling for **S**imulation **O**rchestration in **PY**thon

**R**eally **H**igh-order **A**da**P**tive coupling for **S**imulation **O**rchestration in **PY**thon

This pure Python package demonstrates the application of a high-order code coupling strategy. It enables an accurate resolution of transient multiphysics problems requiring the coupling of existing specialised codes, e.g. fluid flow solver, heat diffusion solver...

The framework is focused on problems of the form:
$$d_t y_i = f_i(t, y_i, u_i)$$
$$u_i = g_i(t, y_1, \cdots, y_N)$$
with $t$ the physical time, $i \in [1,N]$, $N$ the number of subsystems, $y_i$ the state vector of the $i$-th subsystem, $u_i$ its input or *coupling variables*, e.g. prescribed boundary conditions, volumic source term... Many coupled multiphysics models (fluid-structure, conjugate heat transfer) take this form.

The strategy relies on the introduction of approximations of the coupling variables as polynomials of time. Over the course of one coupling step, aach subsystem is integrated with its dedicated solver, using the polynomially approximated evolution of its input $u_i$. At the end of each step, the new coupling variables are computed and the polynomials are updated.

It is possible to perform the coupling in **explicit** or **implicit** form. The latter improves both accuracy and stability, but requires the resolution of a fixed-point problem at each step.
**Dynamic adaptation of the coupling time step** is possible thanks to error estimates that can be directly derived from the polynomial approximations.

The present repository offers an example implementation of the previous strategy, with tutorials in the form of Jupyter Notebooks.

It has been developed at ONERA by Laurent François, based on a study initiated during his [PhD thesis](https://www.theses.fr/2022IPPAX004).

To test this package, run `python setup.py develop` to register it as a development package.

Contact: laurent.francois@onera.fr