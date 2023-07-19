# This code is an implementation of the model presented in Sætra et al. 2023.

import dolfin as df

default_init_parameters = {"alpha_i": "0.4",
                           "alpha_e": "0.2",
                           "Na_i": "15.474585472970270",
                           "K_i": "99.892102216365814",
                           "Cl_i": "5.363687689337043",
                           "HCO3_i": "11.2",
                           "Na_e": "144.090829054058730",
                           "K_e": "3.215795567266669",
                           "Cl_e": "133.272624621326230",
                           "HCO3_e": "8.5",
                           "phi_i": "-0.085861202415139",
                           "phi_e": "0.0"}


class ModelBase():
    """ Modeling electrodiffusive, osmotic, and hydrostatic
        interplay in astrocyte networks. """

    def __init__(self, model_v, mesh, L, t_PDE, options: dict = None):

        self.model_v = model_v      # ('M1', 'M2' or 'M3') model version
        self.mesh = mesh            # mesh
        self.L = L                  # length of domain (m)
        self.t_PDE = t_PDE          # time constant
        self.N_ions = 4             # number of ions
        self.N_comparts = 2         # number of compartments
        self.stim_protocol = None

        # set parameters and initial conditions
        self.set_initial_conditions()
        self.set_parameters()

        return

    def set_parameters(self):
        """ Set the model's physical parameters """

        # physical model parameters
        temperature = df.Constant(310.15)  # temperature [K]
        F = df.Constant(96485.3)           # Faraday's constant [C/mol]
        R = df.Constant(8.314)             # gas constant [J/(mol*K)]

        # diffusion coefficients
        D_Na = df.Constant(1.33e-9)        # sodium [m^2/s]
        D_K = df.Constant(1.96e-9)         # potassium [m^2/s]
        D_Cl = df.Constant(2.03e-9)        # chloride [m^2/s]
        D = [D_Na, D_K, D_Cl]

        # valences
        z_Na = df.Constant(1.0)            # sodium (Na)
        z_K = df.Constant(1.0)             # potassium (K)
        z_Cl = df.Constant(-1.0)           # chloride (Cl)
        z_0 = df.Constant(0.0) 
        z_NBC = df.Constant(-1.0)
        z_HCO3 = df.Constant(-1.0)            # immobile ions (a)
        # x_0 will be calculated at the end of this function
        z = [z_Na, z_K, z_Cl, z_0, z_NBC, z_HCO3]

        # tortuosities
        lambda_i = df.Constant(3.2)
        lambda_e = df.Constant(1.6)
        lambdas = [lambda_i, lambda_e]

        # membrane parameters
        gamma_m = df.Constant(8.0e6)       # area volume ratio [1/m]
        K_m = df.Constant(2.294e3)         # membrane stiffness [Pa]
        g_Na = df.Constant(1.0)            # sodium conductance [S/m^2]
        g_Cl = df.Constant(0.5)            # chloride conductance [S/m^2]
        g_K = df.Constant(16.96)           # potassium conductance [S/m^2]
        g_NBC = df:Constant(7.6e-1)        # NBC contransport conductance [S/m^2]
        rho_pump = df.Constant(1.12e-6)    # max pump rate [mol/(m^2s)]
        P_Nai = df.Constant(10.0)          # pump threshold - Na_i [mol/m^3]
        P_Ke = df.Constant(1.5)            # pump threshold - K_e [mol/m^3]
        eta_m = df.Constant(8.14e-14)      # membrare water permeab. [m/(Pa*s)]

        # compartmental fluid flow parameters
        kappa_i = df.Constant(1.8375e-14)  # ICS water permeability [m^4/(N*s)]
        kappa_e = df.Constant(1.8375e-14)  # ECS water permeability [m^4/(N*s)]
        kappa = [kappa_i, kappa_e]
        eps_r = df.Constant(84.6)          # relative permittivity
        eps_zero = df.Constant(8.85e-12)   # vacuum permittivity [F/m]
        zeta = df.Constant(-22.8e-3)       # zeta-potential [V]
        mu = df.Constant(6.4e-4)           # water viscosity [Pa*s]

        # initial hydrostatic pressure across the membrane
        p_m_init = df.Constant(1.0e3)     # [Pa]

        # gather physical parameters
        params = {'temperature': temperature, 'F': F, 'R': R,
                  'gamma_m': gamma_m, 'K_m': K_m,
                  'D': D, 'z': z,
                  'lambdas': lambdas,
                  'g_Na': g_Na, 'g_Cl': g_Cl, 'g_K': g_K, 'g_NBC': g_NBC,
                  'rho_pump': rho_pump, 'P_Ke': P_Ke, 'P_Nai': P_Nai,
                  'eta_m': eta_m, 'kappa': kappa,
                  'eps_r': eps_r, 'eps_zero': eps_zero,
                  'zeta': zeta, 'mu': mu,
                  'p_m_init': p_m_init}

        # set physical parameters
        self.params = params
        # calculate and set immobile ions
        self.set_immobile_ions()

        return

    def set_immobile_ions(self):
        """ Calculate and set amount of immobile ions """

        # get parameters
        R = self.params['R']
        T = self.params['temperature']
        z_Na = self.params['z'][0]
        z_K = self.params['z'][1]
        z_Cl = self.params['z'][2]
        z_0 = self.params['z'][3]
        z_HCO3 = self.params['z'][5]
        p_m_init = float(self.params['p_m_init'])

        # get initial conditions
        alpha_i = float(self.alpha_i_init)
        alpha_e = float(self.alpha_e_init)
        Na_e = float(self.Na_e_init)
        K_e = float(self.K_e_init)
        Cl_e = float(self.Cl_e_init)
        HCO3_e_init = float(self.HCO3_e_init)         
        Na_i = float(self.Na_i_init)
        K_i = float(self.K_i_init)
        Cl_i = float(self.Cl_i_init)
        HCO3_i_init = float(self.HCO3_i_init) 


        # calculate valence and amount of immobile ions
        z_0 = (Na_e*z_Na + K_e*z_K + Cl_e*z_Cl + HCO3_e*z_HCO3 - Na_i*z_Na -
               K_i*z_K - Cl_i*z_Cl - HCO3_i*z_HCO3) / (p_m_init/(R*T) + Na_e
                                       + K_e + Cl_e + HCO3_e - Na_i - K_i - Cl_i - HCO3_e)
        a_e = - (Na_e*z_Na + K_e*z_K + Cl_e*z_Cl + HCO3_e*z_HCO3) * alpha_e / z_0
        a_i = - (Na_i*z_Na + K_i*z_K + Cl_i*z_Cl + HCO3_i*z_HCO3) * alpha_i / z_0

        # set valence of immobile ions
        self.params['z'][3] = df.Constant(z_0)

        # set amount of immobile ions (mol/m^3)
        a = [a_i, a_e]
        self.params['a'] = a

        return

    def set_initial_conditions(self, options: dict = None):
        """ Set initial conditions """

        in_options = default_init_parameters.copy()
        if options is not None:
            in_options.update(options)

        # volume fractions
        self.alpha_i_init = in_options['alpha_i']  # ICS
        self.alpha_e_init = in_options['alpha_e']  # ECS

        # Ion concentrations
        self.Na_i_init = in_options['Na_i']    # ICS Na [mol/m^3]
        self.K_i_init = in_options['K_i']      # ICS K [mol/m^3]
        self.Cl_i_init = in_options['Cl_i']    # ICS Cl [mol/m^3]
        self.HCO3_i_init = in_options['HCO3_i'] # ICS HCO3 [mol/m^3]

        self.Na_e_init = in_options['Na_e']    # ECS Na [mol/m^3]
        self.K_e_init = in_options['K_e']      # ECS K [mol/m^3]
        self.Cl_e_init = in_options['Cl_e']    # ECS Cl [mol/m^3]
        self.HCO3_e_init = in_options['HCO3_e'] # ECS HCO3 [mol/m^3]

        # electrical potentials
        self.phi_i_init = in_options['phi_i']  # ICS [V]
        self.phi_e_init = in_options['phi_e']  # ECS [V]

        # extracellular hydrostatic pressure
        self.p_e_init = '0.0'
        # Lagrange multiplier
        self.c_init = '0.0'

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

    def j_leak_Na(self, phi_m, E_Na):
        """ Sodium leak flux """

        # get parameters
        F = self.params['F']
        z_Na = self.params['z'][0]
        g_Na = self.params['g_Na']

        # define and return flux (mol/(m^2s))
        j = g_Na*(phi_m - E_Na)/(F*z_Na)

        return j

    def j_leak_Cl(self, phi_m, E_Cl):
        """ Chloride leak flux """

        # get parameters
        F = self.params['F']
        z_Cl = self.params['z'][2]
        g_Cl = self.params['g_Cl']

        # define and return flux (mol/(m^2*s))
        j = g_Cl*(phi_m - E_Cl)/(F*z_Cl)

        return j

    def j_Kir(self, phi_m, E_K, K_e):
        """ Potassium inward rectifying (Kir) flux """

        # get parameters
        F = self.params['F']
        R = self.params['R']
        temperature = self.params['temperature']
        z_K = self.params['z'][1]
        g_K = self.params['g_K']
        K_i_init = float(self.K_i_init)
        K_e_init = float(self.K_e_init)

        # set conductance
        E_K_init = R*temperature/(F*z_K)*df.ln(K_e_init/K_i_init)
        dphi = phi_m - E_K
        A = 1 + df.exp(18.4/42.4)                                  # shorthand
        B = 1 + df.exp(-(0.1186 + E_K_init)/0.0441)                # shorthand
        C = 1 + df.exp((dphi + 0.0185)/0.0425)                    # shorthand
        D = 1 + df.exp(-(0.1186 + phi_m)/0.0441)                   # shorthand
        f_Kir = df.sqrt(K_e/K_e_init)*(A*B)/(C*D)

        # define and return flux (mol/(m^2*s))
        j = g_K*f_Kir*(phi_m - E_K)/(F*z_K)

        return j

    def j_pump(self, K_e, Na_i):
        """ Na/K-pump flux"""

        # get parameters
        rho_pump = self.params['rho_pump']
        P_Nai = self.params['P_Nai']
        P_Ke = self.params['P_Ke']

        # define and return flux (mol/(m^2s))
        j = rho_pump*((Na_i**1.5 / (Na_i**1.5 + P_Nai**1.5))
                      * (K_e / (K_e + P_Ke)))

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
        E_Na = R*temperature/(F*z_Na)*df.ln(Na_e/Na_i)  # sodium    - (V)
        E_K = R*temperature/(F*z_K)*df.ln(K_e/K_i)      # potassium - (V)
        E_Cl = R*temperature/(F*z_Cl)*df.ln(Cl_e/Cl_i)  # chloride  - (V)

        # membrane fluxes
        j_leak_Na = self.j_leak_Na(phi_m, E_Na)
        j_leak_Cl = self.j_leak_Cl(phi_m, E_Cl)
        j_Kir = self.j_Kir(phi_m, E_K, K_e)
        j_pump = self.j_pump(K_e, Na_i)

        # total transmembrane ion fluxes
        j_Na = j_leak_Na + 3.0*j_pump           # sodium    - (mol/(m^2s))
        j_K = j_Kir - 2.0*j_pump                # potassium - (mol/(m^2s))
        j_Cl = j_leak_Cl                        # chloride  - (mol/(m^2s))

        j_m = [j_Na, j_K, j_Cl]

        # set the membrane fluxes
        self.membrane_fluxes = j_m

        return

    def set_input_fluxes(self, w):
        """ Set the input fluxes. """

        # total input fluxes
        j_in_K = df.Constant(0)
        j_in_Na = df.Constant(0)
        j_in_Cl = df.Constant(0)

        j_in = [j_in_Na, j_in_K, j_in_Cl]

        # set the input fluxes (mol/(m^2s))
        self.input_fluxes = j_in

        return