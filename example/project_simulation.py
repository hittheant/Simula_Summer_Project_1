import dolfin as df
import os
import inspect

# set path to solver
from project_flow_models import modelMC1, Solver
from plotter import Plotter


def run_model(model_v, j_in, Tstop, stim_start, stim_end, stim_protocol):
    """
    Arguments:
        model_v (str): model version
        j_in (float): constant input in input zone (mol/(m^2s))
        Tstop (float): end time (s)
        stim_start (float): stimulus onset (s)
        stim_end (float): stimulus offset (s)
        stim_protocol (str): stimulus protocol
    """

    # mesh
    N = 400                                  # mesh size
    L = 3.0e-4                               # m (300 um)
    mesh = df.IntervalMesh(N, 0, L)          # create mesh

    # time variables
    dt_value = 1e-3                          # time step (s)

    # model setup
    t_PDE = df.Constant(0.0)  # time constant

    # model initialization
    class_name = f"Model{model_v}"
    if class_name in globals() and inspect.isclass(globals()[class_name]):
        model_type = globals()[class_name]
        model = model_type(model_v, mesh, L, t_PDE, j_in, stim_start, stim_end, stim_protocol)
    else:
        raise Exception("Invalid model version")

    # check that directory for results (data) exists, if not create
    path_data = 'results/data/' + model_v + '/'

    if not os.path.isdir(path_data):
        os.makedirs(path_data)

    S = Solver(model, dt_value, Tstop)

    S.solve_system(path_results=path_data)

    return model, path_data


if __name__ == '__main__':
    model_v = "MC1"             # Model (hypothesis) number
    j_in = 1.0e-6               # input constant (mol/(m^2s))
    Tstop = 30                  # duration of simulation (s)
    stim_start = 10             # stimulus onset (s)
    stim_end = 20               # stimulus offset (s)
    stim_protocol = 'constant'  # stimulues protocol ('constant', 'slow', or 'ultraslow')

    # run model
    model, path_data = run_model(model_v, j_in, Tstop, stim_start, stim_end, stim_protocol)

    # create plotter object for visualizing results
    P = Plotter(model, path_data)

    # check that directory for figures exists, if not create
    path_figs = 'results/figures/'
    if not os.path.isdir(path_figs):
        os.makedirs(path_figs)

    # plot figures
    P.timeplot(path_figs, model_v, Tstop)
    P.spaceplot(path_figs, model_v, stim_end)
