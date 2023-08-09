import dolfin as df
from .model_base import ModelBase


class ModelMC2(ModelBase):
    """ Model setup with input zone in the middle of the domain. """

    def __init__(self, model_v, mesh, L, t_PDE, j_in_const, stim_start, stim_end, stim_protocol='constant'):
        ModelBase.__init__(self, model_v, mesh, L, t_PDE)

        self.stim_start = stim_start  # time of input onset (s)
        self.stim_end = stim_end  # time of input offset (s)
        self.j_in_const = j_in_const  # constant input in input zone (mol/(m^2s))
        self.stim_protocol = stim_protocol  # stimulus protocol

        self.set_initial_conditions()
        self.set_parameters()
        self.set_immobile_ions()

    def set_parameters(self):
        ModelBase.set_parameters(self)
        g_KCC1 = df.Constant(7e-1)
        self.params['g_KCC1'] = g_KCC1

    def j_in(self, t):
        """ Constant input flux. """

        L_in = 0.1 * self.L  # length of input zone (m)
        L1 = self.L / 2 - L_in / 2
        L2 = self.L / 2 + L_in / 2

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

        k_dec = df.Constant(2.9e-8)  # decay factor for [K]_e (m/s)

        j = - k_dec * (K_e - float(self.K_e_init))

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

    def j_KCC1(self):
        """ KCC1 co-transporter flux """

        # get parameters
        F = self.params['F']
        R = self.params['R']
        temperature = self.params['temperature']
        g_KCC1 = self.params['g_KCC1']
        K_i_init = float(self.K_i_init)
        K_e_init = float(self.K_e_init)
        Cl_i_init = float(self.Cl_i_init)
        Cl_e_init = float(self.Cl_e_init)

        # define and return flux (mol/(m^2*s))

        j_KCC1 = (g_KCC1 / F) * (R * temperature / F) * df.ln((K_e_init / K_i_init) * (Cl_e_init / Cl_i_init))

        return j_KCC1

    def set_membrane_fluxes(self, w):
        """ Set the transmembrane ion fluxes. """

        # get parameters
        F = self.params['F']
        R = self.params['R']
        temperature = self.params['temperature']
        z_Na = self.params['z'][0]
        z_K = self.params['z'][1]
        z_Cl = self.params['z'][2]

        # split unknowns
        alpha_i, Na_i, Na_e, K_i, K_e, Cl_i, Cl_e, \
            phi_i, phi_e, p_e, c = df.split(w)

        # membrane potential
        phi_m = phi_i - phi_e

        # reversal potentials
        E_Na = R * temperature / (F * z_Na) * df.ln(Na_e / Na_i)  # sodium    - (V)
        E_K = R * temperature / (F * z_K) * df.ln(K_e / K_i)  # potassium - (V)
        E_Cl = R * temperature / (F * z_Cl) * df.ln(Cl_e / Cl_i)  # chloride  - (V)

        # membrane fluxes
        j_leak_Na = self.j_leak_Na(phi_m, E_Na)
        j_leak_Cl = self.j_leak_Cl(phi_m, E_Cl)
        j_leak_K = self.j_leak_K(phi_m, E_K)
        j_pump = self.j_pump(K_e, Na_i)
        j_KCC1 = self.j_KCC1()

        # total transmembrane ion fluxes
        j_Na = j_leak_Na + 3.0 * j_pump  # sodium    - (mol/(m^2s))
        j_K = j_leak_K - 2.0 * j_pump + j_KCC1  # potassium - (mol/(m^2s))
        j_Cl = j_leak_Cl + j_KCC1  # chloride  - (mol/(m^2s))

        j_m = [j_Na, j_K, j_Cl]

        # set the membrane fluxes
        self.membrane_fluxes = j_m

        return


