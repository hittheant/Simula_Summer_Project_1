import dolfin as df
import os

# set path to solver
from ffian import flow_model, zero_flow_model
from plotter import Plotter


def run_model(model_v, j_in, Tstop, stim_start, stim_end):
    """
    Arguments:
        model_v (str): model version
        j_in (float): constant input in input zone (mol/(m^2s))
        Tstop (float): end time (s)
        stim_start (float): stimuli onset (s)
        stim_end (float): stimuli offset (s)
    """

    # mesh
    N = 400                                  # mesh size
    L = 3.0e-4                               # m (300 um)
    mesh = df.IntervalMesh(N, 0, L)          # create mesh

    # time variables
    dt_value = 1e-3                          # time step (s)

    # model setup
    t_PDE = df.Constant(0.0)                 # time constant

    if model_v == "M0":
        model = zero_flow_model.Model(mesh, L, t_PDE, j_in, stim_start, stim_end)
    else:
        model = flow_model.Model(model_v, mesh, L, t_PDE, j_in, stim_start, stim_end)

    # check that directory for results (data) exists, if not create
    path_data = 'results/data/' + model_v + '/'

    if not os.path.isdir(path_data):
        os.makedirs(path_data)

    # solve system
    if model_v == "M0":
        S = zero_flow_model.Solver(model, dt_value, Tstop)
    else:
        S = flow_model.Solver(model, dt_value, Tstop)

    S.solve_system(path_results=path_data)

    return model, path_data


if __name__ == '__main__':


    model_v = 'M3'      # model version ('M1', 'M2', 'M3', or 'M0')
    j_in = 1.0e-6       # constant input in input zone (mol/(m^2s))
    Tstop = 30          # duration of simulation (s)
    stim_start = 10     # stimuli onset (s)
    stim_end = 20       # stimuli offset (s)

    # run model
    model, path_data = run_model(model_v, j_in, Tstop, stim_start, stim_end)

    # create plotter object for visualizing results
    P = Plotter(model, path_data)

    # check that directory for figures exists, if not create
    path_figs = 'results/figures/'
    if not os.path.isdir(path_figs):
        os.makedirs(path_figs)

    # plot figures
    P.timeplot(path_figs, model_v, Tstop)
    P.spaceplot(path_figs, model_v, 20)
