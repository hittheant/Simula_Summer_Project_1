# This code is an implementation of the model presented in Halnes et al. 2013
# with minor adjustments.

from dolfin import *
import ufl
import sys
import numpy as np

default_init_parameters = {"Na_i":"15.474585472970270", \
                           "K_i":"99.892102216365814", \
                           "Cl_i":"5.363687689337043", \
                           "Na_e":"144.090829054058730", 
                           "K_e":"3.215795567266669", 
                           "Cl_e":"133.272624621326230", 
                           "phi_i":"-0.085861202415139", 
                           "phi_e":"0.0"}
 
class ModelBase():
    """ Modeling of astrocytic ion concentration dynamics. """

    def __init__(self, mesh, L, t_PDE, options:dict=None):
        
        self.mesh = mesh            # mesh
        self.L = L                  # length of domain (m) 
        self.t_PDE = t_PDE          # time constant (for updating source and boundary terms)
        self.N_ions = 3             # number of ions
        self.N_comparts = 2         # number of compartments
        
        # set parameters and initial conditions
        self.set_initial_conditions(options)
        self.set_parameters()
        
        return

    def set_parameters(self):
        """ Set the model's physical parameters """
        
        # physical model parameters
        temperature = Constant(310.15) # temperature - (K)
        F = Constant(96485.3)          # Faraday's constant - (C/mol)
        R = Constant(8.314)            # gas constant - (J/(mol*K))

        # ion specific parameters
        D_Na = Constant(1.33e-9)       # diffusion coefficient - sodium (m^2/s)
        D_K = Constant(1.96e-9)        # diffusion coefficient - potassium (m^2/s)
        D_Cl = Constant(2.03e-9)       # diffusion coefficient - chloride (m^2/s)
        D = [D_Na, D_K, D_Cl]

        z_Na = Constant(1.0)           # valence - sodium (Na)
        z_K = Constant(1.0)            # valence - potassium (K)
        z_Cl = Constant(-1.0)          # valence - chloride (Cl)
        z_0 = Constant(0.0)            # valence - immobile ions (X) - will be calculated later
        z = [z_Na, z_K, z_Cl, z_0]

        # volume fractions
        alpha_i = Constant(0.4)
        alpha_e = Constant(0.2)
        
        # tortuosities
        lambda_i = Constant(3.2)
        lambda_e = Constant(1.6)
        lambdas = [lambda_i, lambda_e]

        # membrane parameters
        gamma_m = Constant(8.0e6)      # membrane area per unit volume of tissue - (1/m)
        C_m = Constant(1.0e-2)         # capacitance - (F/m^2)
        K_m = Constant(2.294e3)        # membrane stiffness - (Pa)
        g_Na = Constant(1.0)           # sodium conductance - (S/m^2)
        g_Cl = Constant(0.5)           # chloride conductance - (S/m^2)
        g_K = Constant(16.96)          # baseline potassium conductance - (S/m^2)
        rho_pump = Constant(1.12e-6)   # max pump rate - (mol/(m^2s))
        P_Nai = Constant(10.0)         # pump threshold - Na_i (mol/m^3)
        P_Ke = Constant(1.5)           # pump threshold - K_e (mol/m^3)

        # initial hydrostatic pressure across the membrane
        p_m_init = Constant(1.0e3)      # (Pa)

        # gather physical parameters
        params = {'temperature':temperature, 'F':F, 'R':R,
                  'alpha_i':alpha_i, 'alpha_e':alpha_e,
                  'gamma_m':gamma_m, 'C_m':C_m, 'K_m':K_m,
                  'D':D, 'z':z,
                  'lambdas':lambdas,
                  'g_Na':g_Na, 'g_Cl':g_Cl, 'g_K':g_K,
                  'rho_pump':rho_pump, 'P_Ke':P_Ke, 'P_Nai':P_Nai,
                  'p_m_init':p_m_init}

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
        alpha_i = self.params['alpha_i']
        alpha_e = self.params['alpha_e']
        p_m_init = float(self.params['p_m_init'])
    
        # get initial condition
        Na_e = float(self.Na_e_init)
        K_e = float(self.K_e_init)
        Cl_e = float(self.Cl_e_init)
        Na_i = float(self.Na_i_init)
        K_i = float(self.K_i_init)
        Cl_i = float(self.Cl_i_init)

        # calculate valence and amount of immobile ions
        z_0 = (Na_e*z_Na + K_e*z_K + Cl_e*z_Cl - Na_i*z_Na - K_i*z_K - Cl_i*z_Cl) / \
              (p_m_init/(R*T) + Na_e + K_e + Cl_e - Na_i - K_i - Cl_i)
        a_e = - (Na_e*z_Na + K_e*z_K + Cl_e*z_Cl) * alpha_e / z_0
        a_i = - (Na_i*z_Na + K_i*z_K + Cl_i*z_Cl) * alpha_i / z_0

        # set valence of immobile ions (mol/m^3)
        self.params['z'][3] = Constant(z_0)
        
        # set amount of immobile ions (mol/m^3)
        a = [a_i, a_e]
        self.params['a'] = a
        
        return

    def set_initial_conditions(self, options:dict=None):
        """ set initial conditions """
        
        in_options = default_init_parameters.copy()
        if options is not None:
            in_options.update(options)
        
        self.Na_i_init = in_options['Na_i']   # ICS sodium concentration - (mol/m^3)
        self.K_i_init = in_options['K_i']     # ICS potassium concentration - (mol/m^3)
        self.Cl_i_init = in_options['Cl_i']   # ICS chloride concentration - (mol/m^3)

        self.Na_e_init = in_options['Na_e']   # ECS sodium concentration - (mol/m^3)
        self.K_e_init = in_options['K_e']     # ECS potassium concentration - (mol/m^3)
        self.Cl_e_init = in_options['Cl_e']   # ECS chloride concentration - (mol/m^3)

        self.phi_i_init = in_options['phi_i'] # ICS potential (V)
        self.phi_e_init = in_options['phi_e'] # ECS potential (V)

        self.c_init = '0.0'                   # Lagrange multiplier

        self.inits_PDE = Constant((self.Na_i_init, \
                                   self.Na_e_init, \
                                   self.K_i_init, \
                                   self.K_e_init, \
                                   self.Cl_i_init, \
                                   self.Cl_e_init, \
                                   self.phi_i_init, \
                                   self.phi_e_init, \
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
        E_K_init = R*temperature/(F*z_K)*ln(K_e_init/K_i_init)
        dphi = phi_m - E_K
        A = 1 + exp(18.4/42.4)                                  # shorthand
        B = 1 + exp(-(0.1186 + E_K_init)/0.0441)                # shorthand
        C = 1 + exp((dphi + 0.0185)/0.0425)                     # shorthand
        D = 1 + exp(-(0.1186 + phi_m)/0.0441)                   # shorthand
        f_Kir = sqrt(K_e/K_e_init)*(A*B)/(C*D)

        # define and return flux (mol/(m^2*s)) 
        j = g_K*f_Kir*(phi_m - E_K)/(F*z_K)
        
        return j

    def j_pump(self, K_e, Na_i):
        """ Na/K-pump flux"""
        
        # get parameters
        rho_pump = self.params['rho_pump']
        P_Nai = self.params['P_Nai']
        P_Ke = self.params['P_Ke']

        # define and return flux 
        j = rho_pump*((Na_i**1.5 / (Na_i**1.5 + P_Nai**1.5)) * (K_e / (K_e + P_Ke))) # mol/(m^2s)
        
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
        Na_i, Na_e, K_i, K_e, Cl_i, Cl_e, \
                phi_i, phi_e, c = split(w)

        # membrane potential
        phi_m = phi_i - phi_e

        # reversal potentials 
        E_Na = R*temperature/(F*z_Na)*ln(Na_e/Na_i) # sodium    - (V)
        E_K = R*temperature/(F*z_K)*ln(K_e/K_i)     # potassium - (V)
        E_Cl = R*temperature/(F*z_Cl)*ln(Cl_e/Cl_i) # chloride  - (V)

        # membrane fluxes 
        j_leak_Na = self.j_leak_Na(phi_m, E_Na)
        j_leak_Cl = self.j_leak_Cl(phi_m, E_Cl)
        j_Kir = self.j_Kir(phi_m, E_K, K_e)
        j_pump = self.j_pump(K_e, Na_i)

        # total transmembrane ion fluxes
        j_Na = j_leak_Na + 3.0*j_pump               # sodium    - (mol/(m^2s))
        j_K = j_Kir - 2.0*j_pump                    # potassium - (mol/(m^2s))
        j_Cl = j_leak_Cl                            # chloride  - (mol/(m^2s))

        j_m = [j_Na, j_K, j_Cl]
        
        # set the membrane fluxes
        self.membrane_fluxes = j_m
        
        return

    def set_input_fluxes(self, w):
        """ Set the input fluxes """

        # total input fluxes
        j_in_K = Constant(0)
        j_in_Na = Constant(0)
        j_in_Cl = Constant(0)
       
        j_in = [j_in_Na, j_in_K, j_in_Cl]
        
        # set the input fluxes (mol/(m^2s))
        self.input_fluxes = j_in

        return 
