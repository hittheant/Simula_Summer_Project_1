# ffian: Fluid Flow In Astrocyte Networks

The role of astrocyte networks in brain volume homeostasis and waste
clearance has not received enough attention from the neuroscience community.
However, recent research efforts indicate that glial cells are crucial for fluid flow
through brain tissue, contributing to clearance and maintenance of brain volume.
We examine the role of various glial cotransporters in the spatial and temporal
changes of the intra- and extracellular volume fractions and fluid dynamics via
computational modelling. The model is incorporated within the Kirchhoff-Nernst-
Planck electrodiffusive framework and takes into account ionic electrodiffusion and
fluid dynamics. Our research shows that all model configurations demonstrate similar
fluid fluxes, except those involving HCO−
3 dynamics. The model configuration that
included the NBC cotransporter was observed to have the greatest intracellular total
volume-weighted fluid velocity of 16 μm/s.

ffian is an implementation of the KNP continuity equations for a 
one-dimensional system containing two compartments: 
one representing an astrocyte network (ICS) and one representing the
extracellular space (ECS). `ffian.project_flow_models` takes transmembrane- and
compartmental fluid flow into account and predicts the evolution in time
and distribution in space of the volume fractions, 
ion concentrations (Na<sup>+</sup>, K<sup>+</sup>, Cl<sup>-</sup>), 
electrical potentials, and hydrostatic
pressures in the ICS and ECS. Each model in the project_flow_models package includes a different combination of cotransporters, with the 'model_base' representing only leak channels.
The fluid model is presented in Sætra et al. 2023, [Neural activity induces strongly coupled electro-chemo-mechanical interactions and fluid flow in astrocyte networks and extracellular space – a computational study](https://www.biorxiv.org/content/10.1101/2023.03.06.531247v1).

## Previous code

This code has been adapted from the repository at [https://martejulie.github.io/fluid-flow-in-astrocyte-networks](https://martejulie.github.io/fluid-flow-in-astrocyte-networks).

## Installation

### Conda

> **Warning**
> If you want to run the examples, and use conda to install `ffian`, you need to have `texlive-core` installed on your system.

Start by cloning into the repository:

``` console
$ git clone https://github.com/martejulie/fluid-flow-in-astrocyte-networks.git
$ cd ffian
```

Then, using the ``environment.yml`` file in the root of the repository, you can call:



``` console
$ conda env update --file environment.yml --name your_environment
```

Next, you can now activate your environment by running::

``` console
$ conda activate your_environment
```

Finally, install `ffian` inside your `conda` environment using `pip`: 

``` console
$ python3 -m pip install .
```


### Source

To install the `ffian`-library from source, navigate to the root of the repository and
run the following commands from the command line:
```bash
python3 -m pip install python/. --upgrade
```

`ffian` requires `fenics-dolfin`, `numpy`, and `matplotlib`.

## Run simulations

The example folder includes code showing how to run the simulations. Use project_simulation.py to run simulations with models with expanded cotransporters.
To reproduce the results in the paper, see 'Plot Simulation Data' jupyter notebook.
