import dolfin as df
from .model_base import ModelBase

# default_init_parameters = {"HCO3_i": "11.2",
#                            "HCO3_e": "8.5"}
default_init_parameters = {"alpha_i": "0.4",
                           "alpha_e": "0.2",
                           "Na_i": "14.873097896755867",
                           "K_i": "98.52044712996694",
                           "Cl_i": "3.6910413660406522",
                           "HCO3_i": "8.52594263933312",
                           "Na_e": "149.17327134716948",
                           "K_e": "3.039545331969974",
                           "Cl_e": "136.6179172679198",
                           "HCO3_e": "13.848114721334",
                           "phi_i": "-0.08468340395057906",
                           "phi_e": "0.0"}


class ModelMC5(ModelBase):
    """ Model setup with input zone in the middle of the domain. """
    def __init__(self, model_v, mesh, L, t_PDE, j_in_const, stim_start, stim_end, stim_protocol='constant'):
        ModelBase.__init__(self, model_v, mesh, L, t_PDE)
        self.N_ions = 4

        # Stimulation Protocol
        self.stim_start = stim_start  # time of input onset (s)
        self.stim_end = stim_end  # time of input offset (s)
        self.j_in_const = j_in_const  # constant input in input zone (mol/(m^2s))
        self.stim_protocol = stim_protocol  # stimulus protocol

        self.set_initial_conditions()
        self.set_parameters()
        # calculate and set immobile ions
        self.set_immobile_ions()

    def set_parameters(self):
        ModelBase.set_parameters(self)

        z_NBC = df.Constant(-1.0)
        z_HCO3 = df.Constant(-1.0)
        z_0 = self.params['z'][3]
        self.params['z'][3] = z_HCO3
        self.params['z'].extend([z_0])
        self.params['z_NBC'] = z_NBC
        """ Set the model's physical parameters """

        g_NBC = df.Constant(7.6e-1)
        self.params['g_NBC'] = g_NBC
        g_NKCC1 = df.Constant(2.0e-2)  # Flux parameter of NKCC1 cotransporter [ohm^-1 cm^-2]
        self.params['g_NKCC1'] = g_NKCC1

        D_HCO3 = df.Constant(1.09e-9)
        self.params['D'].extend([D_HCO3])

        return

    def set_immobile_ions(self):
        """ Calculate and set amount of immobile ions """

        # get parameters
        R = self.params['R']
        T = self.params['temperature']
        z_Na = self.params['z'][0]
        z_K = self.params['z'][1]
        z_Cl = self.params['z'][2]
        z_0 = self.params['z'][4]
        z_HCO3 = self.params['z'][3]
        p_m_init = float(self.params['p_m_init'])

        # get initial conditions
        alpha_i = float(self.alpha_i_init)
        alpha_e = float(self.alpha_e_init)
        Na_e = float(self.Na_e_init)
        K_e = float(self.K_e_init)
        Cl_e = float(self.Cl_e_init)
        HCO3_e = float(self.HCO3_e_init)
        Na_i = float(self.Na_i_init)
        K_i = float(self.K_i_init)
        Cl_i = float(self.Cl_i_init)
        HCO3_i = float(self.HCO3_i_init)


        # calculate valence and amount of immobile ions
        z_0 = (Na_e*z_Na + K_e*z_K + Cl_e*z_Cl + HCO3_e*z_HCO3 - Na_i*z_Na -
               K_i*z_K - Cl_i*z_Cl - HCO3_i*z_HCO3) / (p_m_init/(R*T) + Na_e
                                       + K_e + Cl_e + HCO3_e - Na_i - K_i - Cl_i - HCO3_e)
        a_e = - (Na_e*z_Na + K_e*z_K + Cl_e*z_Cl + HCO3_e*z_HCO3) * alpha_e / z_0
        a_i = - (Na_i*z_Na + K_i*z_K + Cl_i*z_Cl + HCO3_i*z_HCO3) * alpha_i / z_0

        # set valence of immobile ions
        self.params['z'][4] = df.Constant(z_0)

        # set amount of immobile ions (mol/m^3)
        a = [a_i, a_e]
        self.params['a'] = a

        return

    def set_initial_conditions(self, options: dict = None):
        """ Set initial conditions """
        in_options = default_init_parameters.copy()
        ModelBase.set_initial_conditions(self, in_options)

        self.HCO3_i_init = in_options['HCO3_i'] # ICS HCO3 [mol/m^3]
        self.HCO3_e_init = in_options['HCO3_e'] # ECS HCO3 [mol/m^3]

        self.inits_PDE = df.Constant((self.alpha_i_init,
                                     self.Na_i_init,
                                     self.Na_e_init,
                                     self.K_i_init,
                                     self.K_e_init,
                                     self.Cl_i_init,
                                     self.Cl_e_init,
                                     self.HCO3_i_init,
                                     self.HCO3_e_init,
                                     self.phi_i_init,
                                     self.phi_e_init,
                                     self.p_e_init,
                                     self.c_init))

        return

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
        """ Set the transmembrane ion fluxes """

        # get parameters
        F = self.params['F']
        R = self.params['R']
        temperature = self.params['temperature']
        z_Na = self.params['z'][0]
        z_K = self.params['z'][1]
        z_Cl = self.params['z'][2]
        z_NBC = self.params['z_NBC']

        g_Na = self.params['g_Na']
        g_K = self.params['g_K']
        g_Cl = self.params['g_Cl']
        g_NBC = self.params['g_NBC']


        # split unknowns
        alpha_i, Na_i, Na_e, K_i, K_e, Cl_i, Cl_e, HCO3_i, HCO3_e, \
            phi_i, phi_e, p_e, c = df.split(w)

        # membrane potential
        phi_m = phi_i - phi_e

        # reversal potentials
        E_Na = R*temperature/(F*z_Na)*df.ln(Na_e/Na_i)  # sodium    - (V)
        E_K = R*temperature/(F*z_K)*df.ln(K_e/K_i)      # potassium - (V)
        E_Cl = R*temperature/(F*z_Cl)*df.ln(Cl_e/Cl_i)  # chloride  - (V)
        E_NBC = R*temperature/(F*z_NBC)*df.ln((Na_e*HCO3_e**2)/(Na_i*HCO3_i**2))

        # phi_m = (g_Na*E_Na + g_K*E_K + g_Cl*E_Cl + g_NBC*E_NBC - j_NaKATPase*F)/(g_Na + g_K + g_Cl + g_NBC)

        # membrane fluxes
        j_leak_Na = self.j_leak_Na(phi_m, E_Na)
        j_leak_Cl = self.j_leak_Cl(phi_m, E_Cl)
        j_leak_K = self.j_leak_K(phi_m, E_K)
        j_pump = self.j_pump(K_e, Na_i)
        j_NBC = self.j_NBC(phi_m, E_NBC)
        j_NKCC1 = self.j_NKCC1(Na_i, Na_e, K_i, K_e, Cl_i, Cl_e)

        # total transmembrane ion fluxes
        j_Na = j_leak_Na + 3.0 * j_pump - j_NKCC1 - j_NBC       # sodium    - (mol/(m^2s))
        j_K = j_leak_K - 2.0 * j_pump - j_NKCC1                 # potassium - (mol/(m^2s))
        j_Cl = j_leak_Cl - 2.0 * j_NKCC1                        # chloride  - (mol/(m^2s))
        j_HCO3 = -2.0 * j_NBC

        j_m = [j_Na, j_K, j_Cl, j_HCO3]

        # set the membrane fluxes
        self.membrane_fluxes = j_m

        return

    def j_NBC(self, phi_m, E_NBC):
        # get parameters
        F = self.params['F']
        g_NBC = self.params['g_NBC']

        # set conductance
        j_NBC = g_NBC/F*(phi_m - E_NBC)

        return j_NBC

    def j_NKCC1(self, Na_i, Na_e, K_i, K_e, Cl_i, Cl_e):
        " " " Ion fluxes through NKCC1. " " "

        # # split unknowns into components

        g_NKCC1 = self.params['g_NKCC1']
        F = self.params['F']
        R = self.params['R']
        T = self.params['temperature']

        j = (g_NKCC1/F)*(R*T/F)*df.ln((Na_e/Na_i)*(K_e/K_i)*(Cl_i/Cl_e)**2)

        return j