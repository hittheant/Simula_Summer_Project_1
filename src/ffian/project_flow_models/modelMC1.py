import dolfin as df
from .model_base import ModelBase


class ModelMC1(ModelBase):
    """ Model setup with input zone in the middle of the domain. """

    def __init__(self, model_v, mesh, L, t_PDE, j_in_const, stim_start, stim_end, stim_protocol='constant'):
        ModelBase.__init__(self, model_v, mesh, L, t_PDE)

        # Stimulation Protocol
        self.stim_start = stim_start  # time of input onset (s)
        self.stim_end = stim_end  # time of input offset (s)
        self.j_in_const = j_in_const  # constant input in input zone (mol/(m^2s))
        self.stim_protocol = stim_protocol  # stimulus protocol

        # set parameters and initial conditions
        self.set_initial_conditions()
        self.set_parameters()
        # calculate and set immobile ions
        self.set_immobile_ions()

    def j_in(self, t):
        """ Constant input flux. """

        L_in = 0.1*self.L             # length of input zone (m)
        L1 = self.L/2-L_in/2
        L2 = self.L/2+L_in/2

        if self.stim_protocol == 'constant':
            j = df.Expression('j_in*(x[0] > L1)*(x[0] < L2)*(t >= tS)*(t <= tE)',
                              L1=L1, L2=L2, j_in=self.j_in_const, t=t,
                              tS=self.stim_start, tE=self.stim_end, degree=1)
        elif self.stim_protocol == 'slow':
             j = df.Expression('(x[0] > L1)*(x[0] < L2)*(t >= tS)*(t <= tE)*(j_in/2*sin(2*pi*t) + j_in/2)',
                               L1=L1, L2=L2, j_in=self.j_in_const, t=t, \
                               tS=self.stim_start, tE=self.stim_end, degree=1)
        elif self.stim_protocol == 'ultraslow':
             j = df.Expression('(x[0] > L1)*(x[0] < L2)*(t >= tS)*(t <= tE)*(j_in/2*sin(2*pi*t*0.05) + j_in/2)',
                               L1=L1, L2=L2, j_in=self.j_in_const, t=t, \
                               tS=self.stim_start, tE=self.stim_end, degree=1)

        return j

    def j_dec(self, K_e):
        """ Decay flux proportional to [K]_e. """

        k_dec = df.Constant(2.9e-8)   # decay factor for [K]_e (m/s)

        j = - k_dec*(K_e - float(self.K_e_init))

        return j

    def set_input_fluxes(self, w):
        """ Set input fluxes. """

        # split unknowns
        alpha_i, \
            Na_i, Na_e, K_i, K_e, Cl_i, Cl_e, \
            phi_i, phi_e, p_e, c = df.split(w)

        # input/decay
        j_in = self.j_in(self.t_PDE)
        j_dec = self.j_dec(K_e)

        # total input fluxes
        j_in_K = j_in + j_dec
        j_in_Na = - j_in - j_dec
        j_in_Cl = df.Constant(0)

        j_in_k = [j_in_Na, j_in_K, j_in_Cl]

        # set the input fluxes
        self.input_fluxes = j_in_k

        return