import dolfin as df
from .model_base import ModelBase

default_init_parameters = {"alpha_i": "0.4",
                           "alpha_e": "0.2",
                           "Na_i": "15.510234830675616",
                           "K_i": "99.19015487718889",
                           "Cl_i": "5.064077397202017",
                           "Na_e": "146.5802367864952",
                           "K_e": "3.018983797774815",
                           "Cl_e": "133.87184520559637",
                           "phi_i": "-0.08466809466757282",
                           "phi_e": "0.0"}


class ModelMC4(ModelBase):
    def __init__(self, model_v, mesh, L, t_PDE, j_in_const, stim_start, stim_end, stim_protocol='constant'):
        # initializes the ModelMC4 object, initializing the inherited attributes from ModelBase
        ModelBase.__init__(self, model_v, mesh, L, t_PDE)

        self.stim_start = stim_start         # time of input onset (s)
        self.stim_end = stim_end             # time of input offset (s)
        self.j_in_const = j_in_const         # constant input in input zone (mol/(m^2s))
        self.stim_protocol = stim_protocol   # stimulus protocol

        self.set_initial_conditions()
        self.set_parameters()
        self.set_immobile_ions()

    def set_parameters(self):
        ModelBase.set_parameters(self)

        g_NKCC1 = df.Constant(2.0e-2)  # Flux parameter of NKCC1 cotransporter [ohm^-1 cm^-2]
        self.params['g_NKCC1'] = g_NKCC1

        return

    def set_initial_conditions(self, options: dict = None):
        in_options = default_init_parameters.copy()
        ModelBase.set_initial_conditions(self, in_options)

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
        """ Decay flux. """
        k_dec = df.Constant(2.9e-8)
        j = -k_dec*(K_e-float(self.K_e_init))
        return j

    def set_input_fluxes(self, w):
        """ Set the input fluxes. """
        # split unknowns into components
        alpha_i, Na_i, Na_e, K_i, K_e, Cl_i, Cl_e, phi_i, phi_e, p_e, c = df.split(w)

        # input/decay
        j_in = self.j_in(self.t_PDE)
        j_dec = self.j_dec(K_e)

        # total input fluxes
        j_in_K = j_in + j_dec
        j_in_Na = -j_in - j_dec
        j_in_Cl = df.Constant(0.0)

        j_in_k = [j_in_Na, j_in_K, j_in_Cl]

        # set input fluxes
        self.input_fluxes = j_in_k

    def j_NKCC1(self, w):
        " " " Ion fluxes through NKCC1. " " "

        # split unknowns into components
        alpha_i, Na_i, Na_e, K_i, K_e, Cl_i, Cl_e, phi_i, phi_e, p_e, c = df.split(w)

        g_nkcc1 = self.params['g_NKCC1']
        F = self.params['F']
        R = self.params['R']
        T = self.params['T']

        j = (g_nkcc1/F)*(R*T/F)*df.ln((Na_e/Na_i)*(K_e/K_i)*(Cl_i/Cl_e)**2)

        return j


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
    j_NKCC1 = self.j_NKCC1(w)

    # total transmembrane ion fluxes
    j_Na = j_leak_Na + 3.0 * j_pump - j_NKCC1  # sodium    - (mol/(m^2s))
    j_K = j_leak_K - 2.0 * j_pump - j_NKCC1  # potassium - (mol/(m^2s))
    j_Cl = j_leak_Cl - 2.0 * j_NKCC1 # chloride  - (mol/(m^2s))

    j_m = [j_Na, j_K, j_Cl]

    # set the membrane fluxes
    self.membrane_fluxes = j_m

    return