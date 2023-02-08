import pytest
import numpy as np
from dolfin import *
from ffian.flow_model import ModelBase
from ffian.flow_model import Model 
from ffian.flow_model import Solver

def project_to_function_space(mesh, u):
    """ Project u onto function space """

    CG1 = FiniteElement('CG', mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, CG1)
    f = project(u, V)

    return f

def run_model_base(model_v):
    """ Test that no ions disappear. """

    # model version
    model_v = 'M1'

    # mesh
    N = 400                                  # mesh size
    L = 3.0e-4                               # m (300 um)
    mesh = IntervalMesh(N, 0, L)             # create mesh

    # time variables
    dt_value = 1.0e-3                        # time step (s)
    Tstop = 60                               # end time (s)

    # model
    t_PDE = Constant(0.0)                        # time constant
    model = ModelBase(model_v, mesh, L, t_PDE)   # model
    
    # solve system
    S = Solver(model, dt_value, Tstop)
    S.solve_system()

    # check solution
    w = S.w

    alpha_i, Na_i, Na_e, K_i, K_e, Cl_i, Cl_e, \
        phi_i, phi_e, p_e, c = w.split()
    
    alpha_e = 0.6 - alpha_i

    EPS = 1e-6

    a = model.params['a']
    a_i = a[0]
    a_e = a[1]
    z = model.params['z']
    z_Na = z[0]
    z_K = z[1]
    z_Cl = z[2]
    z_X = z[3]
    
    q_i_ = (Na_i*z_Na + K_i*z_K + Cl_i*z_Cl + a_i/alpha_i*z_X)*alpha_i
    q_e_ = (Na_e*z_Na + K_e*z_K + Cl_e*z_Cl + a_e/alpha_e*z_X)*alpha_e

    q_e = project_to_function_space(mesh, q_e_)
    q_i = project_to_function_space(mesh, q_i_)

    for point in np.linspace(0,L,50):
        assert abs(float(q_e(point))) < EPS
        assert abs(float(q_i(point))) < EPS

def test_model_base_M1():

    run_model_base('M1')

    return 
    
def test_model_base_M2():

    run_model_base('M2')

    return 
    
def test_model_base_M3():

    run_model_base('M3')

    return 

def test_stimulus():
    """ Test that no ions disappear. """

    # model version
    model_v = 'M1'

    # mesh
    N = 400                                  # mesh size
    L = 3.0e-4                               # m (300 um)
    mesh = IntervalMesh(N, 0, L)             # create mesh

    # time variables
    dt_value = 1.0e-3                        # time step (s)
    Tstop = 60                               # end time (s)

    # stimulus
    j_in = 8.0e-7                            # constatn input in input zone (mol/(m^2s))

    # model 
    t_PDE = Constant(0.0)                   # time constant
    model = Model(model_v, mesh, L, t_PDE, j_in, stim_start=10, stim_end=50)    # model
    
    # solve system
    S = Solver(model, dt_value, Tstop)
    S.solve_system()

    # test that no charge dissappear
    w = S.w

    alpha_i, Na_i, Na_e, K_i, K_e, Cl_i, Cl_e, \
        phi_i, phi_e, p_e, c = w.split()
    alpha_e = 0.6 - alpha_i

    EPS = 1e-3

    a = model.params['a']
    a_i = a[0]
    a_e = a[1]
    z = model.params['z']
    z_Na = z[0]
    z_K = z[1]
    z_Cl = z[2]
    z_X = z[3]
    
    q_i_ = (Na_i*z_Na + K_i*z_K + Cl_i*z_Cl + a_i/alpha_i*z_X)*alpha_i
    q_e_ = (Na_e*z_Na + K_e*z_K + Cl_e*z_Cl + a_e/alpha_e*z_X)*alpha_e

    q_e = project_to_function_space(mesh, q_e_)
    q_i = project_to_function_space(mesh, q_i_)

    for point in np.linspace(0,L,50):
        assert abs(float(q_e(point))) < EPS
        assert abs(float(q_i(point))) < EPS
