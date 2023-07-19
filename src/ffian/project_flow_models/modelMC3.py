import dolfin as df
from .model_base import ModelBase


class ModelMC3(ModelMC1):
    """ Model setup with input zone in the middle of the domain. """

    def __init__(self, model_v, mesh, L, t_PDE, j_in_const, stim_start, stim_end, stim_protocol='constant'):
        ModelBase.__init__(self, model_v, mesh, L, t_PDE)
        self.stim_start = stim_start         # time of input onset (s)
        self.stim_end = stim_end             # time of input offset (s)
        self.j_in_const = j_in_const         # constant input in input zone (mol/(m^2s))
        self.stim_protocol = stim_protocol   # stimulus protocol

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
            Na_i, Na_e, K_i, K_e, Cl_i, Cl_e, HCO3_i, HCO3_e, \
            phi_i, phi_e, p_e, c = df.split(w)

        # input/decay
        j_in = self.j_in(self.t_PDE)
        j_dec = self.j_dec(K_e)

        # total input fluxes
        j_in_K = j_in + j_dec
        j_in_Na = - j_in - j_dec
        j_in_Cl = df.Constant(0)
        j_in_HCO3 = df.Constant(0)

        j_in_k = [j_in_Na, j_in_K, j_in_Cl, j_in_HCO3]

        # set the input fluxes
        self.input_fluxes = j_in_k

        return

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
        E_Na = R*temperature/(F*z_Na)*df.ln(Na_e/Na_i)  # sodium    - (V)
        E_K = R*temperature/(F*z_K)*df.ln(K_e/K_i)      # potassium - (V)
        E_Cl = R*temperature/(F*z_Cl)*df.ln(Cl_e/Cl_i)  # chloride  - (V)

        # membrane fluxes
        j_leak_Na = self.j_leak_Na(phi_m, E_Na)
        j_leak_Cl = self.j_leak_Cl(phi_m, E_Cl)
        j_Kir = self.j_Kir(phi_m, E_K, K_e)
        j_pump = self.j_pump(K_e, Na_i)
        j_NBC = self.j_NBC(self, K_e, Na_i, HCO3_e, HCO3_i)

        # total transmembrane ion fluxes
        j_Na = j_leak_Na + 3.0*j_pump           # sodium    - (mol/(m^2s))
        j_K = j_Kir - 2.0*j_pump                # potassium - (mol/(m^2s))
        j_Cl = j_leak_Cl                        # chloride  - (mol/(m^2s))
        j_HCO3 = -2*j_NBC                    

        j_m = [j_Na, j_K, j_Cl, j_HCO3]

        # set the membrane fluxes
        self.membrane_fluxes = j_m

        return

        def j_NBC(self, K_e, Na_i, HCO3_e, HCO3_i):

        # get parameters
        F = self.params['F']
        R = self.params['R']
        temperature = self.params['temperature']
        z_Na = self.params['z'][0]
        z_K = self.params['z'][1]
        z_Cl = self.params['z'][2]
        z_NBC = self.params['z'][4]

        g_Na = self.params['g_Na']
        g_K = self.params['g_K']
        g_Cl = self.params['g_Cl']
        g_NBC = self.params['g_NBC']

        j_NaKATPase = j_pump(self, K_e, Na_i)

        # set conductance
        E_NBC = R*temperature/(F*z_NBC)*df.ln((Na_e*HCO3_e**2)/(Na_i*HCO3_i**2)) # NBC contransport reversal potential

        V_ m = (g_Na*E_Na + g_K*E_K + g_Cl*E_Cl + g_NBC*E_NBC - j_NaKATPase*F)/(g_Na + g_K + g_Cl + g_NBC)

        j_NBC = g_NBC/F*(V_m - E_NBC)

        return j_NBC