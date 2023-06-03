#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:58:53 2021

Model for the 1D heating of a single slab with uniform and constant properties

@author: laurent
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from rhapsopy.coupling import ExceptionWhichMayDisappearWhenLoweringDeltaT

ncalls = 0

def setupFiniteVolumeMesh(xfaces, meshoptions=None):
    """ Setup 1D spatial mash for finite volume, based on the positions of the faces of each cell """
    if meshoptions is None:
        meshoptions={}
    meshoptions['faceX'] = xfaces
    meshoptions['cellX'] = 0.5*(xfaces[1:]+xfaces[0:-1]) # center of each cell
    meshoptions['dxBetweenCellCenters'] = np.diff(meshoptions['cellX']) # gap between each consecutive cell-centers
    meshoptions['cellSize'] = np.diff(xfaces) # size of each cells
    assert not any(meshoptions['cellSize']==0.), 'some cells are of size 0...'
    assert not any(meshoptions['cellSize']<0.), 'some cells are of negative size...'
    assert not any(meshoptions['dxBetweenCellCenters']==0.), 'some cells have the same centers...'
    assert np.max(meshoptions['cellSize'])/np.min(meshoptions['cellSize']) < 1e10, 'cell sizes extrema are too different'
    # conveniency attributes for backward-compatibility  wtih finite-difference results post-processing
    meshoptions['x']  = meshoptions['cellX']
    meshoptions['dx'] = meshoptions['dxBetweenCellCenters']
    return meshoptions


def getTs(coupling, T2, T1, t):
    """ Computes the surface temperature Ts such that the heat flux is
    continuous across the interface """
    a1 = coupling['a1'] # lbda1 / (xs - x1)
    a2 = coupling['a2'] # lbda2 / (xs - x2)
    phi_cor = 0 #coupling['phi_cor'](t) # phi_correctif_2_to_1
    # Ts_inerte = ( phi_cor + T1*a1 - T2*a2 ) / ( a1 - a2 ) # there cannot be a singularity, unless one conductivity is negative...
    Ts_inerte = (T1+T2)/2

    if 'forcedTs' in coupling.keys():
      print('forcing Ts !!!!!')
      return coupling['forcedTs'](t)

    # ONLY COMPATIBLE WITH A BALANCE TYPE COUPLING (for one side at least)
    phi_surf = lambda x: coupling['phi_surf'](t,x)
    if 1: # Newton loop to find the surface temperature
      resid = lambda x: x - ( phi_cor - phi_surf(x) + T1*a1 - T2*a2 ) / ( a1 - a2 )
      # resid = lambda x: x - ( phi_cor - phi_surf(x) + T1*(T1>0)*a1 - T2*(T2>0)*a2 ) / ( a1 - a2 )
      fprime = lambda x: np.imag(resid(x+1e-50*1j))*1e50
      # import pdb; pdb.set_trace()
      try:
        out = scipy.optimize.newton(func=resid, x0=Ts_inerte, fprime=fprime, args=(),
                                    tol=1e-30, maxiter=100, fprime2=None, x1=None,
                                    rtol=1e-14, full_output=True, disp=True)
        Ts = out[0]
      except RuntimeError as e:
        raise ExceptionWhichMayDisappearWhenLoweringDeltaT('Ts did not converge')
        print(e)
        import matplotlib.pyplot as plt
        xtest = np.linspace(1e-3, 10, 1000)
        residuals = resid(xtest)
        plt.semilogy(xtest, abs(residuals))
        plt.xlabel('Ts')
        plt.ylabel("residual")
        plt.title('Nonlinear problem for Ts')
        plt.grid()
        plt.show()
        raise e
        raise ExceptionWhichMayDisappearWhenLoweringDeltaT('Ts did not converge')
    else: # fixed-point
      Ts_old=np.inf*np.ones_like(Ts_inerte)
      Ts=Ts_inerte
      i_iter=0
      # print('phi_surf(Ts_inerte)=',phi_surf(Ts_inerte))
      try:
        tol = 1e-14
        # print('it {}: Ts={}'.format(i_iter,Ts))
        while np.max(np.abs(Ts-Ts_old)/np.abs(tol*Ts+tol))>1:
          if i_iter>50:
            raise Exception('Ts did not converge')
          Ts_old = Ts
          Ts = ( phi_cor - phi_surf(Ts_old) + T1*a1 - T2*a2 ) / ( a1 - a2 )
          i_iter+=1
          # print('it {}: Ts={}'.format(i_iter,Ts))
      except ValueError as e:
        import pdb; pdb.set_trace()
        print(e)
        raise e
      # if np.isscalar(Ts):
      #   print("      Fixed-point for Ts converged in {} iterations (Ts={:.4e})".format(i_iter,Ts))
      # else:
      #   print("      Fixed-point for Ts converged in {} iterations (Ts[0]={:.4e})".format(i_iter,Ts[0]))
      # print('  --> Ts={:.2e}'.format(Ts))
    return Ts


def heatModelfunODE(t,x,options,bDebug=False, imex_mode=-1, sideOfCoupling=None):
    """
    Gives the time derivative of the discretized temperature field in the solid
    Uses ghost cells/points for the boundary conditions
    """
    if imex_mode != -1:
      imex_formul = options['imex_formul'] # 1
    else:
      imex_formul = -99
    
    if imex_mode==1: # remove inner diffusion
      dudt_full = heatModelfunODE(t=t, x=x, options=options, bDebug=bDebug,
                                  imex_mode=-1, sideOfCoupling=sideOfCoupling)
      dudt_imex = heatModelfunODE(t=t, x=x, options=options, bDebug=bDebug,
                                  imex_mode=0, sideOfCoupling=sideOfCoupling)
      dudt_imex2 = dudt_full - dudt_imex # simpler to formulate it that way !
      return dudt_imex2
    
    global ncalls # number of function calls for debug purposes
    ncalls += 1
    if len(x.shape)==1: #compatibility with non-vectorized calls
        x = x[:,np.newaxis]
        bTransposed = True
    else:
        bTransposed = False
        raise Exception('to simplify debugging, do not use the vectorised call')
      
    # get material properties
    D    = options['solid']['D'] # thermal diffusivity
    lbda = options['solid']['lbda'] # thermal conductivity

    # create the time derivatives array
    xtype = x.dtype # to enable complex-step
    time_deriv = np.zeros(np.shape(x), dtype=xtype)

    ## MESH DATA
    # cell-centered finite-volume approach
    xfaces    = options['solid']['mesh']['faceX'] # face coordinates
    xcells    = options['solid']['mesh']['cellX'] # cell centers
    center_dx = options['solid']['mesh']['dxBetweenCellCenters']
    cell_size = options['solid']['mesh']['cellSize']# size of each cell

    ## BOUNDARY CONDITIONS
    # compute ghost values for the boundary conditions
    for side in ('right', 'left'):
        if side=='right':
            index_bnd = -1
        else:
            index_bnd = 0

        if options['solid']['BCs'][side]['type']=='dirichlet':
            # T at face is prescribed
            xghost = options['solid']['BCs'][side]['Timposed'](t)
            if (imex_mode==0) and (imex_formul==1): # remove coupling
              if sideOfCoupling==side:
                # Kazemi implicit part is obtained by setting the interface temperature to 0
                xghost = 0.

        elif options['solid']['BCs'][side]['type']=='neumann':
            # imposed heat flux: flux =+lbda*(dT/dx)
            dTdx_bc = options['solid']['BCs'][side]['flux'](t)/lbda
            if imex_mode==0 and (imex_formul==1): # remove coupling
              if sideOfCoupling==side:
                # Kazemi implicit part is obtained by setting the interface flux to 0
                dTdx_bc = 0.
            xghost = x[index_bnd,:]  + (xfaces[index_bnd] - xcells[index_bnd]) * dTdx_bc      

        elif options['solid']['BCs'][side]['type']=='balance':
          # Ts is defined by an algebraic equation (flux equality on both sides)
          # --> equivalent to having Ts as a discrete variable
          xghost = getTs(options['coupling'][side],
                         T2=options['coupling'][side]['T2'](t),
                         T1=x[index_bnd,:],
                         t=t)
          if imex_mode!=-1:
            raise Exception("Balance mode is not allowed with imex mode. Yes, that's a seemingly arbitrary choice :)")
            
        else:
            raise Exception('unknown BC type {}'.format(options['solid']['BCs'][side]['type']))

        # affect to the corresponding ghost point
        if side=='right':
            xghostR = np.copy(xghost)
        else:
            xghostL = np.copy(xghost)

    # 2 - compute temperature gradient at faces
    bigger_shape = (np.size(x,0)+1,np.size(x,1))
    dTdx = np.zeros(bigger_shape, dtype=xtype) # dTdx[i] = (T[i]-T[i-1])/dx at the faces
    dTdx[1:-1,:] = (x[1:,:]-x[0:-1,:])/center_dx[:,None]
    dTdx[0,:]    = (x[0,:]-xghostL)  / (xcells[0]-xfaces[0])
    dTdx[-1,:]   = (xghostR-x[-1,:]) / (xfaces[-1]-xcells[-1])

    # compute fluxes
    F = - D*dTdx # centered 2nd order scheme

    # add corrective fluxes #TODO: check sign
    F[-1,:]  = F[-1,:] + options['coupling']['right']['phi_cor'](t)
    F[0,:]   = F[0,:]  + options['coupling']['left']['phi_cor'](t)
      
    if (imex_mode==0) and (imex_formul==2): # remove coupling
      if sideOfCoupling=='right':
        F[-1,:] = 0.
      elif sideOfCoupling=='left':
        F[0,:]  = 0.
      else:
        raise Exception('issue')

    # finite-volume scheme
    #      du/dt    = -( F(i+1/2) - F(i-1/2)  ) / dx
    time_deriv[:,:] = -( F[1:,:]  - F[0:-1,:] ) / cell_size[:,None]

    # # add source term
    # rhoC = lbda / D
    # source_term = options['solid']['source'](t,xcells)
    # time_deriv[:,:] = time_deriv[:,:] + (1/rhoC) * source_term[:,np.newaxis]

    if bDebug:
        plt.figure()
        plt.plot(xcells, np.real(x[:,0]) )
        plt.title('T')

        plt.figure()
        plt.plot(xcells, np.real(time_deriv[:,0]) , marker='+')
        plt.title('dT/dt with finite volumes')
        plt.show()
        raise Exception('debug plots drawn')

    if bTransposed:
        time_deriv = time_deriv.ravel()

    return time_deriv

def getEnergy(x,options):
    """ Compute the thermal energy, assuming uniform temperature in each cell,
    and constant thermal properties """
    if len(x.shape)==1: # compatibility with non-vectorized calls
        x = x[:,np.newaxis]
        bTransposed = True
    else:
        bTransposed = False

    rhoC = options['solid']['lbda']/options['solid']['D'] # rho * cp
    E = rhoC * np.sum( (x.T*options['solid']['mesh']['cellSize']).T, axis=0)
    # E = integral(left to right) integral(0 K to T) cp * dT * dx
    if bTransposed:
        E = E.ravel()
    return E

def postProcess(out, options1, options2):
    out.options1 = options1
    out.options2 = options2

    nx1 = out.options1['y0'].size
    out.y1, out.y2 = out.y[:nx1,:], out.y[nx1:,:]

    out.E1 = getEnergy(out.y1, out.options1)
    out.E2 = getEnergy(out.y2, out.options2)
    out.Etot = out.E1 + out.E2
    out.Ts = getTs(coupling=out.options1['coupling']['right'],
            T1=out.y1[-1,:],
            T2=out.y2[0,:],
            t=out.t)
    return out

if __name__=='__main__':
    # Various test cases
    options={}
    options['solid'] = {
              'mesh':{},
               'BCs':{
                 'right':{
                         'type': 'neumann',
                         'Timposed': None,
                         'flux': None,
                         },
                 'left':{
                         'type': 'neumann',
                         'Timposed': None,
                         'flux': None,
                        },
              },
              'D': 1.,
              'lbda': 0.5,
             }
    options['solid']['source'] = lambda t,x: 0.*x

    nCase=1 # test case choice

    if nCase==1: # Diffusion with finite volume and dirichlet conditions
        N = 100
        options['solid']['BCs']={
                 'right':{
                         'type': 'dirichlet',
                         'Timposed': lambda t: 1500,
                         'flux': None,
                         },
                 'left':{
                         'type': 'dirichlet',
                         'Timposed': lambda t: 1000.,
                         'flux': None,
                        }
                 }
        # mesh
        xfaces=np.linspace(-1,1,N)
        options['solid']['mesh'] = setupFiniteVolumeMesh(xfaces=xfaces, meshoptions=options['solid']['mesh'])
        y0 = 1300*options['solid']['mesh']['cellX']**2 # initial solution
        t_span = [0,1e3] # time span

    elif nCase==2: # Diffusion with finite volume, zero total flux
        # --> useful to check energy conservation
        N = 100
        options['solid']['BCs']={
                 'right':{
                         'type': 'neumann',
                         'Timposed': None,
                         'flux': lambda t: 0.0,
                         },
                 'left':{
                         'type': 'neumann',
                         'Timposed': None,
                         'flux': lambda t: 0.,
                        }
                 }

        # mesh
        xfaces=np.linspace(-1,1,N)
        options['solid']['mesh'] = setupFiniteVolumeMesh(xfaces=xfaces, meshoptions=options['solid']['mesh'])

        y0 = 100+ 1300.*options['solid']['mesh']['cellX']**2 + 500*options['solid']['mesh']['cellX']
        t_span = [0,1e3]

    elif nCase==3: # Validation with semi-infinite wall and constant surface heat flux
        # --> the surface temperature must be proportional to sqrt(t)
        N = 1000
        options['solid']['BCs']={
                 'right':{
                         'type': 'neumann',
                         'Timposed': None,
                         'flux': lambda t: 1000.0,
                         },
                 'left':{
                         'type': 'neumann',
                         'Timposed': None,
                         'flux': lambda t: 0.,
                        }
                 }

        # mesh
        xfaces=np.logspace(-6,2,N)
        xfaces= xfaces-xfaces[0]
        xfaces = -xfaces[::-1]
        options['solid']['mesh'] = setupFiniteVolumeMesh(xfaces=xfaces, meshoptions=options['solid']['mesh'])

        y0 = 0.*options['solid']['mesh']['cellX']
        t_span = [0,1]

    if 'coupling' not in options.keys():
      options['coupling'] = {}
    if 'right' not in options['coupling'].keys():
      options['coupling']['right'] = {}
    options['coupling']['right']['phi_cor'] = lambda t: 0.

    if 'left' not in options['coupling'].keys():
      options['coupling']['left'] = {}
    options['coupling']['left']['phi_cor'] = lambda t: 0.

    #%% Temporal integration
    import scipy.optimize._numdiff
    # naive brute force approach for the Jacobian estimation
    ncalls=0
    Jac_objfun_full = scipy.optimize._numdiff.approx_derivative(fun=lambda x: heatModelfunODE(t=0.,x=x,options=options), x0=y0,
                                                                method="cs",
                                                                rel_step=1e-8, f0=None,
                                                                bounds=(-np.inf, np.inf), sparsity=None,
                                                                as_linear_operator=False, args=(), kwargs={},)
    print("Assuming a full sparsity pattern, one Jacobian evaluation requires {}, which is 1 + number of variables ({})".format(ncalls, y0.size))

    # tracé du sparsity pattern
    plt.figure()
    plt.spy(Jac_objfun_full)

    # version intelligente
    # on donne la Jacobienne full à scipy, qui détermine comment grouper les perturbations de manière optimale
    # pour pouvoir faire plusieurs perturbations d'un coup lors de la détermination de la Jacobienne par
    # différences finies
    # g = scipy.optimize._numdiff.group_columns(Jac_objfun_full, order=0)
    jacSparsity = 1*(Jac_objfun_full!=0.)
    ncalls = 0 # on remet à zéro le compteur
    Jac_grouped = scipy.optimize._numdiff.approx_derivative(fun=lambda x: heatModelfunODE(t=0.,x=x,options=options), x0=y0, method="cs",
                                                            rel_step=1e-8, f0=None,
                                                            bounds=(-np.inf, np.inf), sparsity=jacSparsity,
                                                            as_linear_operator=False, args=(), kwargs={})
    print("Using the true sparsity pattern, the Jacobian can be evaluated more efficiently in {} function calls".format(ncalls))
    # On peut aussi générer une fonction qui utilise ces perturbations groupées pour déterminer la Jacobienne
    jacfun = lambda t,x,options: scipy.optimize._numdiff.approx_derivative(fun=lambda y: heatModelfunODE(t=t,x=y,options=options), x0=x, method="cs",
                                                            rel_step=1e-8, f0=None,
                                                            bounds=(-np.inf, np.inf), sparsity=jacSparsity,
                                                            as_linear_operator=False, args=(), kwargs={}).todense()

    f1   = heatModelfunODE(t=t_span[0], x=y0, options=options, bDebug=False, imex_mode=0,  sideOfCoupling='left')
    f2   = heatModelfunODE(t=t_span[0], x=y0, options=options, bDebug=False, imex_mode=1,  sideOfCoupling='left')
    ftot = heatModelfunODE(t=t_span[0], x=y0, options=options, bDebug=False, imex_mode=-1, sideOfCoupling=None)
    assert np.allclose(f1+f2, ftot, rtol=1e-13), 'IMEX formulation seems broken'
    
    out = scipy.integrate.solve_ivp(fun=heatModelfunODE, y0=y0, method='Radau', t_eval=None,
                                    t_span=t_span, atol=1e-4, rtol=1e-4,
                                    args=(options,), jac=jacfun)
    sol, time = out.y, out.t
    print('Solution obtained in {} steps'.format(out.t.size))

    #%% Post-processing
    xfaces    = options['solid']['mesh']['faceX']
    xcells    = options['solid']['mesh']['cellX']#0.5*(xfaces[1:]-xfaces[0:-1])
    center_dx = options['solid']['mesh']['dxBetweenCellCenters']
    cell_size = options['solid']['mesh']['cellSize']# size of each cell
    ylims = (0.9*np.min(sol), 1.1*np.max(sol))
    for i in [0, len(time)//2, len(time)-1]: #range(1, len(time), 10):
        plt.figure()
        plt.plot(xcells, sol[:,i])
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('T')
        plt.ylim(ylims)
        plt.title('t={:.2e} s'.format(time[i]))

    plt.figure()
    plt.plot(time**0.5, sol[-1,:], label=r'$T_R$')
    plt.plot(time**0.5, sol[0,:], label=r'$T_L$')
    plt.grid()
    plt.legend()
    plt.xlabel(r'$\sqrt{t}$')
    plt.ylabel('T')

    # Energy
    E = getEnergy(sol, options)
    if E[0]!=0:
        E = E/E[0]
    plt.figure()
    plt.plot(time, E)
    plt.xlabel('t')
    plt.ylabel('E/E(0)')

    #%% Constant heat flux validation
    if nCase==3:
      Ts_theorique = 0 + options['solid']['BCs']['right']['flux'](0.)/options['solid']['lbda'] * (4*options['solid']['D']*time/np.pi)**0.5
      plt.figure()
      plt.plot(time, sol[-1,:], label='numerical') # techniquement, ce n'est pas Ts, mais la dernière cellule est très petite donc ça devrait aller
      plt.plot(time, Ts_theorique, label='analytical',linestyle='--')
      plt.legend()
      plt.grid()
      plt.xlabel('t (s)')
      plt.ylabel('Ts (K)')
      plt.title('Validation of Ts(t) for constant heat flux')