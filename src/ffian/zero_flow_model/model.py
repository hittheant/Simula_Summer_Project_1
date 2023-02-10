import dolfin as df
from .model_base import ModelBase


class Model(ModelBase):
    """ Problem with input zone in the middle. """

    def __init__(self, mesh, L, t_PDE, j_in_const, stim_start, stim_end):
        ModelBase.__init__(self, mesh, L, t_PDE)
        self.stim_start = stim_start   # time of input onset (s)
        self.stim_end = stim_end       # time of input being turned off (s)
        self.j_in_const = j_in_const   # constant input in input zone (mol/(m^2s))

    def j_in(self, t):
        """ Constant input. """

        L_in = 0.1*self.L       # length of input zone (m)
        L1 = self.L/2-L_in/2
        L2 = self.L/2+L_in/2

        j = df.Expression('j_in*(x[0] > L1)*(x[0] < L2)*(t >= tS)*(t <= tE)',
                          L1=L1, L2=L2, j_in=self.j_in_const, t=t,
                          tS=self.stim_start, tE=self.stim_end, degree=1)

        return j

    def j_dec(self, K_e):
        """ Decay flux proportional to [K]_e. """

        k_dec = df.Constant(2.9e-8)  # decay factor for [K]_e (m/s)

        j = - k_dec*(K_e - float(self.K_e_init))

        return j

    def set_input_fluxes(self, w):
        """ Set input fluxes. """

        # split unknowns
        Na_i, Na_e, K_i, K_e, Cl_i, Cl_e, \
            phi_i, phi_e, c = df.split(w)

        # input/output
        j_in = self.j_in(self.t_PDE)
        j_dec = self.j_dec(K_e)

        # total input/output fluxes
        j_in_K = j_in + j_dec
        j_in_Na = - j_in - j_dec
        j_in_Cl = df.Constant(0)

        j_in_k = [j_in_Na, j_in_K, j_in_Cl]

        # set the input fluxes
        self.input_fluxes = j_in_k

        return
