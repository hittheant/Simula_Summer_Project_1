from dolfin import *
import ufl
import sys
import numpy as np

from .model_base import ModelBase

class Model(ModelBase):
    """ Model setup with input zone in the middle of the domain. """
    
    def __init__(self, model_v, mesh, L, t_PDE, j_in_const, stim_start, stim_end):
        ModelBase.__init__(self, model_v, mesh, L, t_PDE)
        self.stim_start = stim_start   # time of input onset (s)
        self.stim_end = stim_end       # time of input offset (s)
        self.j_in_const = j_in_const   # constant input in input zone (mol/(m^2s)) 

    def j_in(self, t):
        """ Constant input flux. """

        L_in = 0.1*self.L       # length of input zone (m)
        L1 = self.L/2-L_in/2
        L2 = self.L/2+L_in/2
        
        j = Expression('j_in*(x[0] > L1)*(x[0] < L2)*(t >= tS)*(t <= tE)',
                       L1=L1, L2=L2, j_in=self.j_in_const, t=t, \
                       tS=self.stim_start, tE=self.stim_end, degree=1)

        return j

    def j_dec(self, K_e):
        """ Decay flux proportional to [K]_e. """

        k_dec = Constant(2.9e-8) # decay factor for extracellular potassium (m/s)

        j = - k_dec*(K_e - float(self.K_e_init))

        return j

    def set_input_fluxes(self, w):
        """ Set input fluxes. """

        # split unknowns 
        alpha_i, \
        Na_i, Na_e, K_i, K_e, Cl_i, Cl_e, \
                phi_i, phi_e, p_e, c = split(w)

        # input/decay
        j_in = self.j_in(self.t_PDE)
        j_dec = self.j_dec(K_e)
        
        # total input fluxes
        j_in_K = j_in + j_dec
        j_in_Na = - j_in - j_dec
        j_in_Cl = Constant(0)
       
        j_in_k = [j_in_Na, j_in_K, j_in_Cl]
        
        # set the input fluxes
        self.input_fluxes = j_in_k

        return 
