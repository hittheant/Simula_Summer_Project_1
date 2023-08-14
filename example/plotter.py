import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import dolfin as df

# set font & text parameters
font = {'family': 'serif',
        'weight': 'bold',
        'size': 16}
plt.rc('font', **font)
plt.rc('text', usetex=True)
mpl.rcParams['image.cmap'] = 'jet'
plt.rc('legend')
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')

# set colors
colormap = cm.viridis
mus = [1, 2, 3, 4, 5, 6]
colorparams = mus
colormap = cm.viridis
normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))

b0 = '#7fc97fff'
b1 = '#beaed4ff'
b2 = '#fdc086ff'
b3 = '#ffff99ff'

c2 = colormap(normalize(mus[0]))
c1 = colormap(normalize(mus[1]))
c0 = colormap(normalize(mus[2]))
c3 = colormap(normalize(mus[3]))
c4 = colormap(normalize(mus[4]))
c5 = colormap(normalize(mus[5]))
colors = [c1, c2, c3, c4, c5]

# plotting parameters
xlim = [0, 3e-4]  # range of x values (m)
xticks = [0e-3, 0.05e-3, 0.1e-3, 0.15e-3, 0.2e-3, 0.25e-3, 0.3e-3]
xticklabels = ['0', '50', '100', '150', '200', '250', '300']
xlabel_x = '$x$ (um)'
point_time = 1.5e-4

lw = 4.5     # line width
fosi = 18.7  # ylabel font size
fs = 0.9


class Plotter():

    def __init__(self, model, path_data=None):

        self.model = model
        N_ions = self.model.N_ions
        N_comparts = self.model.N_comparts
        self.N_unknowns = N_comparts*(1 + N_ions) + 3

        # initialize mesh
        self.mesh_PDE = df.Mesh()

        # initialize data files
        if path_data is not None:
            self.set_datafile(path_data)

        return

    def set_datafile(self, path_data):
        # file containing data
        self.h5_fname_PDE = path_data + 'PDE/results.h5'

        # create mesh and read data file
        self.hdf5_PDE = df.HDF5File(df.MPI.comm_world, self.h5_fname_PDE, 'r')
        self.hdf5_PDE.read(self.mesh_PDE, '/mesh', False)

        return

    def read_from_file(self, n, i):
        """ get snapshot of solution w[i] at time = n seconds """

        N_comparts = self.model.N_comparts
        N_unknowns = self.N_unknowns

        CG1 = df.FiniteElement('CG', self.mesh_PDE.ufl_cell(), 1)
        R = df.FiniteElement("R", self.mesh_PDE.ufl_cell(), 0)  # element for Lagrange multiplier
        e = [CG1]*(N_comparts - 1) + [CG1]*(N_unknowns - (N_comparts - 1))
        W = df.FunctionSpace(self.mesh_PDE, df.MixedElement(e + [R]))
        u = df.Function(W)

        V_CG1 = df.FunctionSpace(self.mesh_PDE, CG1)
        f = df.Function(V_CG1)

        self.hdf5_PDE.read(u, "/solution/vector_" + str(n))
        df.assign(f, u.split()[i])

        return f

    def project_to_function_space(self, u):
        """ project u onto function space """

        CG1 = df.FiniteElement('CG', self.mesh_PDE.ufl_cell(), 1)
        V = df.FunctionSpace(self.mesh_PDE, CG1)
        f = df.project(u, V)

        return f

    def timeplot(self, path_figs, model_v, Tstop):
        """ Plot input/decay-currents, changes in ECS and ICS ion concentrations,
        changes in ECS and ICS volume fractions,
        changes in transmembrane hydrostatic pressure,
        and membrane potential over time, measured at x = point_time. """

        # get parameters
        K_m = self.model.params['K_m']
        p_m_init = self.model.params['p_m_init']
        alpha_i_init = float(self.model.alpha_i_init)
        alpha_e_init = float(self.model.alpha_e_init)

        # point in space at which to use in timeplots
        point = point_time

        # range of t values
        xlim_T = [0.0, Tstop]

        # list of function values at point
        j_ins = []
        j_decs = []
        Na_is = []
        K_is = []
        Cl_is = []
        HCO3_is = []
        Na_es = []
        K_es = []
        Cl_es = []
        HCO3_es = []
        dalpha_is = []
        dalpha_es = []
        p_ms = []
        phi_ms = []

        for n in range(Tstop+1):

            # get data
            alpha_i = self.read_from_file(n, 0)
            Na_i = self.read_from_file(n, 1)
            Na_e = self.read_from_file(n, 2)
            K_i = self.read_from_file(n, 3)
            K_e = self.read_from_file(n, 4)
            Cl_i = self.read_from_file(n, 5)
            Cl_e = self.read_from_file(n, 6)
            if model_v == 'MC3' or model_v == 'MC5':
                HCO3_i = self.read_from_file(n, 7)
                HCO3_e = self.read_from_file(n, 8)
                phi_i = self.read_from_file(n, 9)
                phi_e = self.read_from_file(n, 10)
            else:
                phi_i = self.read_from_file(n, 7)
                phi_e = self.read_from_file(n, 8)

            # calculate extracellular volume fraction
            alpha_e = 0.6 - alpha_i(point)

            # get input/decay fluxes
            j_in_ = self.model.j_in(n)
            j_dec_ = self.model.j_dec(K_e)
            j_in = self.project_to_function_space(j_in_*1e6)    # convert to umol/(m^2s)
            j_dec = self.project_to_function_space(j_dec_*1e6)  # convert to umol/(m^2s)

            # calculate change in volume fractions
            alpha_i_diff = (alpha_i(point) - alpha_i_init)/alpha_i_init*100
            alpha_e_diff = (alpha_e - alpha_e_init)/alpha_e_init*100

            # calculate transmembrane hydrostatic pressure
            tau = K_m*(alpha_i(point) - alpha_i_init)
            p_m = tau + p_m_init

            # calculate membrane potential
            phi_m = (phi_i(point) - phi_e(point))*1000  # convert to mV

            # append data to lists
            j_ins.append(j_in(point))
            j_decs.append(j_dec(point))
            Na_is.append(Na_i(point))
            K_is.append(K_i(point))
            Cl_is.append(Cl_i(point))
            Na_es.append(Na_e(point))
            K_es.append(K_e(point))
            Cl_es.append(Cl_e(point))
            if model_v == 'MC3' or model_v == 'MC5':
                HCO3_is.append(HCO3_i(point))
                HCO3_es.append(HCO3_e(point))
            dalpha_is.append(alpha_i_diff)
            dalpha_es.append(alpha_e_diff)
            p_ms.append(float(p_m))
            phi_ms.append(phi_m)

        # f = open('final_points.txt', "w")
        print(f"alpha_i final = {dalpha_is[-1]}")
        print(f"alpha_e final = {dalpha_es[-1]}")
        print(f"Na_i final = {Na_is[-1]}")
        print(f"Na_e final = {Na_es[-1]}")
        print(f"K_i final = {K_is[-1]}")
        print(f"K_e final = {K_es[-1]}")
        print(f"Cl_i final = {Cl_is[-1]}")
        print(f"Cl_i final = {Cl_es[-1]}")
        if model_v == 'MC3' or model_v == 'MC5':
            print(f"HCO3_i final = {HCO3_i(point)}")
            print(f"HCO3_e final = {HCO3_e(point)}")
        print(f"phi_i final = {phi_i(point)}")
        print(f"phi_e final = {phi_e(point)}")

        # create plot
        fig = plt.figure(figsize=(11*fs, 15*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(4, 2, 1, xlim=xlim_T)#, ylim=[-0.25, 1.25])
        plt.ylabel(r'$j\mathrm{^K_{input}}$($\mu$mol/(m$^2$s))', fontsize=fosi)
        plt.plot(j_ins, color='k', linestyle='dotted', linewidth=lw)

        ax2 = fig.add_subplot(4, 2, 2, xlim=xlim_T)#, ylim=[-0.25, 1.25])
        plt.ylabel(r'$j\mathrm{^K_{decay}}$($\mu$mol/(m$^2$s))', fontsize=fosi)
        plt.plot(j_decs, color='k', linewidth=lw)

        ax3 = fig.add_subplot(4, 2, 3, xlim=xlim_T)#, ylim=[-15, 15])
        plt.ylabel(r'$\Delta [k]_\mathrm{e}$ (mM)', fontsize=fosi)
        plt.plot(np.array(Na_es)-Na_es[0], color=b0, label=r'Na$^+$', linewidth=lw)
        plt.plot(np.array(K_es)-K_es[0], color=b1, label=r'K$^+$', linestyle='dotted', linewidth=lw)
        plt.plot(np.array(Cl_es)-Cl_es[0], color=b2, label=r'Cl$^-$', linestyle='dashed', linewidth=lw)
        if model_v == 'MC3' or model_v == 'MC5':
            plt.plot(np.array(HCO3_es) - HCO3_es[0], color=b3, label=f'HCO$_3^-$', linestyle='dashdot', linewidth=lw)

        ax4 = fig.add_subplot(4, 2, 4, xlim=xlim_T)#, ylim=[-15, 15])
        plt.ylabel(r'$\Delta [k]_\mathrm{i}$ (mM)', fontsize=fosi)
        plt.plot(np.array(Na_is)-Na_is[0], color=b0, linewidth=lw)
        plt.plot(np.array(K_is)-K_is[0], color=b1, linestyle='dotted', linewidth=lw)
        plt.plot(np.array(Cl_is)-Cl_is[0], color=b2, linestyle='dashed', linewidth=lw)
        if model_v == 'MC3' or model_v == 'MC5':
            plt.plot(np.array(HCO3_is) - HCO3_is[0], color=b3, linestyle='dashdot', linewidth=lw)

        ax5 = fig.add_subplot(4, 2, 5, xlim=xlim_T)#, ylim=[-10, 10])
        plt.ylabel(r'$\Delta \alpha_\mathrm{e}$ (\%) ', fontsize=fosi)
        plt.plot(dalpha_es, color=c0, linewidth=lw)

        ax6 = fig.add_subplot(4, 2, 6, xlim=xlim_T)#, ylim=[-10, 10])
        plt.ylabel(r'$\Delta \alpha_\mathrm{i}$ (\%) ', fontsize=fosi)
        plt.plot(dalpha_is, color=c0, linewidth=lw)

        ax7 = fig.add_subplot(4, 2, 7, xlim=xlim_T)
        plt.ylabel(r'$\Delta(p_\mathrm{i}- p_\mathrm{e})$ (Pa)', fontsize=fosi)
        plt.plot(np.array(p_ms)-float(p_m_init), color=c2, linewidth=lw)
        plt.xlabel(r'time (s)', fontsize=fosi)

        ax8 = fig.add_subplot(4, 2, 8, xlim=xlim_T)#, ylim=[-90, -60])
        plt.ylabel(r'$\phi_\mathrm{m}$ (mV)', fontsize=fosi)
        plt.plot(phi_ms, color=c1, linewidth=lw)
        plt.xlabel(r'time (s)', fontsize=fosi)

        plt.figlegend(bbox_to_anchor=(0.26, 0.76), frameon=True)

        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

        # make pretty
        ax.axis('off')

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{A}', r'\textbf{B}',
                   r'\textbf{C}', r'\textbf{D}',
                   r'\textbf{E}', r'\textbf{F}',
                   r'\textbf{G}', r'\textbf{H}']
        for num, ax in enumerate(axes):
            ax.text(-0.2, 1.0, letters[num], transform=ax.transAxes, size=22, weight='bold')
            # make pretty
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        plt.tight_layout()
        # save figure to file
        fname_res = path_figs + 'timeplot_' + model_v
        plt.savefig(fname_res + '.pdf', format='pdf')
        plt.close()

        return

    def spaceplot(self, path_figs, model_v, n):
        """ Plot spatial profiles of the input/decay-currents,
        changes in ECS and ICS ion concentrations,
        changes in ECS and ICS volume fractions,
        changes in transmembrane hydrostatic pressure,
        and membrane potential at t = n. """

        # get parameters
        K_m = self.model.params['K_m']
        p_m_init = self.model.params['p_m_init']
        alpha_i_init = float(self.model.alpha_i_init)
        alpha_e_init = float(self.model.alpha_e_init)

        # get data
        alpha_i_ = self.read_from_file(n, 0)
        Na_i_ = self.read_from_file(n, 1)
        Na_e_ = self.read_from_file(n, 2)
        K_i_ = self.read_from_file(n, 3)
        K_e_ = self.read_from_file(n, 4)
        Cl_i_ = self.read_from_file(n, 5)
        Cl_e_ = self.read_from_file(n, 6)
        if model_v == 'MC3' or model_v == 'MC5':
            HCO3_i_ = self.read_from_file(n, 7)
            HCO3_e_ = self.read_from_file(n, 8)
            phi_i_ = self.read_from_file(n, 9)
            phi_e_ = self.read_from_file(n, 10)
        else:
            phi_i_ = self.read_from_file(n, 7)
            phi_e_ = self.read_from_file(n, 8)

        # calculate extracellular volume fraction
        alpha_e_ = 0.6 - alpha_i_

        # get input/decay fluxes
        j_in_ = self.model.j_in(n)
        j_dec_ = self.model.j_dec(K_e_)

        # calculate transmembrane hydrostatic pressure
        tau = K_m*(alpha_i_ - alpha_i_init)
        p_m = tau + p_m_init

        # calculate membrane potential
        phi_m_ = (phi_i_ - phi_e_)*1000  # convert to mV

        # changes from baseline
        dalpha_i_ = (alpha_i_ - alpha_i_init)/alpha_i_init*100
        dalpha_e_ = (alpha_e_ - alpha_e_init)/alpha_e_init*100
        dNa_i_ = Na_i_ - float(self.model.Na_i_init)
        dNa_e_ = Na_e_ - float(self.model.Na_e_init)
        dK_i_ = K_i_ - float(self.model.K_i_init)
        dK_e_ = K_e_ - float(self.model.K_e_init)
        dCl_i_ = Cl_i_ - float(self.model.Cl_i_init)
        dCl_e_ = Cl_e_ - float(self.model.Cl_e_init)
        if model_v == 'MC3' or model_v == 'MC5':
            dHCO3_i_ = HCO3_i_ - float(self.model.HCO3_i_init)
            dHCO3_e_ = HCO3_e_ - float(self.model.HCO3_e_init)
        dp_m_ = p_m - p_m_init

        # project to function space
        j_in = self.project_to_function_space(j_in_*1e6)    # convert to umol/(m^2s)
        j_dec = self.project_to_function_space(j_dec_*1e6)  # convert to umol/(m^2s)
        dalpha_i = self.project_to_function_space(dalpha_i_)
        dalpha_e = self.project_to_function_space(dalpha_e_)
        dNa_i = self.project_to_function_space(dNa_i_)
        dNa_e = self.project_to_function_space(dNa_e_)
        dK_i = self.project_to_function_space(dK_i_)
        dK_e = self.project_to_function_space(dK_e_)
        dCl_i = self.project_to_function_space(dCl_i_)
        dCl_e = self.project_to_function_space(dCl_e_)
        if model_v == 'MC3' or model_v == 'MC5':
            dHCO3_i = self.project_to_function_space(dHCO3_i_)
            dHCO3_e = self.project_to_function_space(dHCO3_e_)
        dp_m = self.project_to_function_space(dp_m_)
        phi_m = self.project_to_function_space(phi_m_)

        # create plot
        fig = plt.figure(figsize=(11*fs, 15*fs))
        ax = plt.gca()

        ax1 = fig.add_subplot(4, 2, 1, xlim=xlim)#, ylim=[-0.25, 1.25])
        plt.xticks(xticks, xticklabels)
        plt.ylabel(r'$j\mathrm{^K_{input}}$($\mu$mol/(m$^2$s))', fontsize=fosi)
        df.plot(j_in, color='k', linestyle='dotted', linewidth=lw)

        ax2 = fig.add_subplot(4, 2, 2, xlim=xlim)#, ylim=[-0.25, 1.25])
        plt.xticks(xticks, xticklabels)
        plt.ylabel(r'$j\mathrm{^K_{decay}}$($\mu$mol/(m$^2$s))', fontsize=fosi)
        df.plot(j_dec, color='k', linewidth=lw)

        ax3 = fig.add_subplot(4, 2, 3, xlim=xlim)#, ylim=[-15, 15])
        plt.xticks(xticks, xticklabels)
        plt.ylabel(r'$\Delta [k]_\mathrm{e}$ (mM)', fontsize=fosi)
        df.plot(dNa_e, color=b0, label=r'Na$^+$', linewidth=lw)
        df.plot(dK_e, color=b1, label=r'K$^+$', linestyle='dotted', linewidth=lw)
        df.plot(dCl_e, color=b2, label=r'Cl$^-$', linestyle='dashed', linewidth=lw)
        if model_v == 'MC3' or model_v == 'MC5':
            df.plot(dHCO3_e, color=b3, label=r'HCO$_3^-$', linestyle='dashdot', linewidth=lw)

        ax4 = fig.add_subplot(4, 2, 4, xlim=xlim)#, ylim=[-15, 15])
        plt.xticks(xticks, xticklabels)
        plt.ylabel(r'$\Delta [k]_\mathrm{i}$ (mM)', fontsize=fosi)
        df.plot(dNa_i, color=b0, linewidth=lw)
        df.plot(dK_i, color=b1, linestyle='dotted', linewidth=lw)
        df.plot(dCl_i, color=b2, linestyle='dashed', linewidth=lw)
        if model_v == 'MC3' or model_v == 'MC5':
            df.plot(dHCO3_i, color=b3, linestyle='dashdot', linewidth=lw)

        ax5 = fig.add_subplot(4, 2, 5, xlim=xlim)#, ylim=[-10, 10])
        plt.xticks(xticks, xticklabels)
        plt.ylabel(r'$\Delta \alpha_\mathrm{e}$ (\%) ', fontsize=fosi)
        df.plot(dalpha_e, color=c0, linewidth=lw)

        ax6 = fig.add_subplot(4, 2, 6, xlim=xlim)#, ylim=[-10, 10])
        plt.xticks(xticks, xticklabels)
        plt.ylabel(r'$\Delta \alpha_\mathrm{i}$ (\%) ', fontsize=fosi)
        df.plot(dalpha_i, color=c0, linewidth=lw)

        ax7 = fig.add_subplot(4, 2, 7, xlim=xlim)
        plt.xticks(xticks, xticklabels)
        plt.ylabel(r'$\Delta(p_\mathrm{i}- p_\mathrm{e})$ (Pa)', fontsize=fosi)
        df.plot(dp_m, color=c2, linewidth=lw)
        plt.xlabel(xlabel_x, fontsize=fosi)

        ax8 = fig.add_subplot(4, 2, 8, xlim=xlim)#, ylim=[-90, -60])
        plt.xticks(xticks, xticklabels)
        plt.ylabel(r'$\phi_\mathrm{m}$ (mV)', fontsize=fosi)
        df.plot(phi_m, color=c1, linewidth=lw)
        plt.xlabel(xlabel_x, fontsize=fosi)

        plt.figlegend(bbox_to_anchor=(0.26, 0.76), frameon=True)

        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

        # make pretty
        ax.axis('off')

        # add numbering for the subplots (A, B, C etc)
        letters = [r'\textbf{A}', r'\textbf{B}',
                   r'\textbf{C}', r'\textbf{D}',
                   r'\textbf{E}', r'\textbf{F}',
                   r'\textbf{G}', r'\textbf{H}']
        for num, ax in enumerate(axes):
            ax.text(-0.2, 1.0, letters[num], transform=ax.transAxes, size=22, weight='bold')
            # make pretty
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        plt.tight_layout()
        print("creating space plots")
        # save figure to file
        fname_res = path_figs + 'spaceplot_' + model_v
        plt.savefig(fname_res + '.pdf', format='pdf')
        plt.close()
        print("end")
        return
