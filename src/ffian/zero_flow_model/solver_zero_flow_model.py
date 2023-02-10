import dolfin as df


class Solver():
    """ Class for solving the zero-flow model used in Sætra et al. 2023
        with unknowns w = (k_r, phi_r_) where:

        k_r     - concentration of ion species k in compartment r
        phi_r   - potential in compartment r """

    def __init__(self, model, dt_value, Tstop):
        """ Initialize solver """

        # time variables
        self.dt = df.Constant(dt_value)     # time step
        self.Tstop = Tstop                  # end time

        # get model
        self.model = model                  # model
        self.mesh = model.mesh              # mesh
        self.N_ions = model.N_ions          # number of ions
        self.N_comparts = model.N_comparts  # number of compartments

        # create function spaces
        self.setup_function_spaces()

        # create PDE solver
        self.PDE_solver()

        return

    def setup_function_spaces(self):
        """ Create function spaces for PDE solver """

        N_comparts = self.N_comparts                   # number of compartments
        N_ions = self.N_ions                           # number of ions
        self.N_unknowns = N_comparts*(1 + N_ions) + 1  # number of unknowns

        # define function space
        CG1 = df.FiniteElement('CG', self.mesh.ufl_cell(), 1)
        R = df.FiniteElement("R", self.mesh.ufl_cell(), 0)  # for Lagr. multip.
        elements = [CG1]*(self.N_unknowns - 1) + [R]

        ME = df.MixedElement(elements)                      # mixed element
        self.W = df.FunctionSpace(self.mesh, ME)            # function space

        # initial conditions
        inits_PDE = self.model.inits_PDE
        self.w_ = df.interpolate(inits_PDE, self.W)

        # unknowns (use initial conditions as guess in Newton solver)
        self.w = df.interpolate(inits_PDE, self.W)

        return

    def PDE_solver(self):
        """ Create variational formulation for PDEs """

        # get parameters
        params = self.model.params
        # get number of compartments and ions
        N_comparts = self.model.N_comparts
        N_ions = self.model.N_ions

        # physical parameters
        temperature = params['temperature']  # temperature
        F = params['F']                      # Faraday's constant
        R = params['R']                      # gas constant

        # membrane parameters
        gamma_m = params['gamma_m']         # are to volume ratio

        # ion specific parameters
        z = params['z']                     # valence of ions
        D = params['D']                     # diffusion coefficients

        # compartmental parameters
        alpha_i = params['alpha_i']         # volume fraction - ICS
        alpha_e = params['alpha_e']         # volume fraction - ECS
        lambda_i = params['lambdas'][0]     # turtuosity - ICS
        lambda_e = params['lambdas'][1]     # turtuosity - ECS

        # split function for unknown solution in current step n+1
        ww = df.split(self.w)
        # split function for known solution in previous time step n
        ww_ = df.split(self.w_)
        # define test functions
        vv = df.TestFunctions(self.W)

        # set transmembrane ion fluxes
        self.model.set_membrane_fluxes(self.w)
        # get transmembrane ion fluxes
        j_m = self.model.membrane_fluxes

        # set the input/output ion fluxes
        self.model.set_input_fluxes(self.w)
        # get the input/output ion fluxes
        j_in = self.model.input_fluxes

        # initiate variational formulation
        A_k_i = 0              # for ICS conservation of ions
        A_k_e = 0              # for ECS conservation of ions
        A_phi_i = 0            # for ICS potential
        A_phi_e = 0            # for ECS potential
        A_lag = 0              # for Lagrange multiplier

        # shorthands
        phi_i = ww[-3]             # ICS potential
        v_phi_i = vv[-3]           # test function for ICS potential
        phi_e = ww[-2]             # ECS potential
        v_phi_e = vv[-2]           # test function for ECS potential

        c = ww[-1]                 # Lagrange multiplier
        d = vv[-1]                 # test function for Lagrange multiplier

        # add terms from constraint (Lagrange multiplier)
        A_phi_e += c*v_phi_e*df.dx
        A_lag += phi_e*d*df.dx

        # variational formulations for ions and potentials
        for n in range(N_ions):
            # index for ion n
            index_i = N_comparts*(n + 1) - 2
            index_e = N_comparts*(n + 2) - 3
            # shorthands ICS
            k_i = ww[index_i]          # unknown ion concentration ICS
            k_i_ = ww_[index_i]        # previous ion concentration ICS
            v_k_i = vv[index_i]        # test function ion concentration ICS
            D_i = D[n]/lambda_i**2     # effective diffusion coefficients ICS
            # shorthands ECS
            k_e = ww[index_e]          # unknown ion concentration ECS
            k_e_ = ww_[index_e]        # previous ion concentration ECS
            v_k_e = vv[index_e]        # test function ion concentration ECS
            D_e = D[n]/lambda_e**2     # effective diffusion coefficients ECS

            # ICS ion flux for ion n - (mol/m^2*s)
            j_i = - D_i*(df.grad(k_i)
                        + z[n]*F*k_i/(R*temperature)*df.grad(phi_i))

            # ECS ion flux for ion n - (mol/m^2*s)
            j_e = - D_e*(df.grad(k_e)
                        + z[n]*F*k_e/(R*temperature)*df.grad(phi_e))

            # form for conservation of ion n in ICS
            A_k_i += 1.0/self.dt*df.inner(k_i - k_i_, v_k_i)*df.dx \
                        - df.inner(j_i, df.grad(v_k_i))*df.dx \
                        + (gamma_m/alpha_i)*df.inner(j_m[n], v_k_i)*df.dx

            # form for conservation of ion n in ECS
            A_k_e += 1.0/self.dt*df.inner(k_e - k_e_, v_k_e)*df.dx \
                        - df.inner(j_e, df.grad(v_k_e))*df.dx \
                        - (gamma_m/alpha_e)*df.inner(j_m[n], v_k_e)*df.dx \
                        - (gamma_m/alpha_e)*df.inner(j_in[n], v_k_e)*df.dx

            # add ion specific part to form for ICS potential
            A_phi_i += - df.inner(z[n]*alpha_i*j_i, df.grad(v_phi_i))*df.dx \
                        + gamma_m*df.inner(z[n]*j_m[n], v_phi_i)*df.dx

            # add ion specific part to form for ECS potential
            A_phi_e += - df.inner(z[n]*alpha_e*j_e, df.grad(v_phi_e))*df.dx \
                        - gamma_m*df.inner(z[n]*j_m[n], v_phi_e)*df.dx

        # assemble system
        self.A = A_k_i + A_k_e + A_phi_i + A_phi_e + A_lag

        # initiate solver
        bcs = None
        J = df.derivative(self.A, self.w)
        model = df.NonlinearVariationalProblem(self.A, self.w, bcs, J)
        self.PDE_solver = df.NonlinearVariationalSolver(model)
        prm = self.PDE_solver.parameters

        prm['newton_solver']['absolute_tolerance'] = 1E-14
        prm['newton_solver']['relative_tolerance'] = 1E-10
        prm['newton_solver']['maximum_iterations'] = 10
        prm['newton_solver']['relaxation_parameter'] = 1.0

        return

    def solve_system(self, path_results=False):
        """ Solve PDE system with iterative Newton solver """

        # save results at every second
        eval_int = float(1.0/self.dt)

        # initialize saving of results
        if path_results:
            filename = path_results
            self.initialize_h5_savefile(filename + 'PDE/' + 'results.h5')

        # initiate iteration number
        k = 1

        while (float(self.model.t_PDE) <= self.Tstop):
            print("Current time:", float(self.model.t_PDE))

            # update time and solve PDE system
            self.model.t_PDE.assign(float(self.dt + self.model.t_PDE))
            self.PDE_solver.solve()     # solve
            self.w_.assign(self.w)      # update previous PDE solutions

            # save results every eval_int'th time step
            if (k % eval_int == 0) and path_results:
                self.save_h5()

            # update iteration number
            k += 1

        # close results files
        if path_results:
            self.close_h5()

        return

    def initialize_h5_savefile(self, filename):
        """ initialize h5 file """
        self.h5_idx_PDE = 0
        self.h5_file_PDE = df.HDF5File(self.mesh.mpi_comm(), filename, 'w')
        self.h5_file_PDE.write(self.mesh, '/mesh')
        self.h5_file_PDE.write(self.w, '/solution',  self.h5_idx_PDE)
        return

    def save_h5(self):
        """ save results to h5 file """
        self.h5_idx_PDE += 1
        self.h5_file_PDE.write(self.w, '/solution',  self.h5_idx_PDE)
        return

    def close_h5(self):
        """ close h5 file """
        self.h5_file_PDE.close()
        return
