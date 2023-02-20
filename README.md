# ffian: Fluid Flow In Astrocyte Networks

ffian is an implementation of the KNP continuity equations for a 
one-dimensional system containing two compartments: 
one representing an astrocyte network (ICS) and one representing the
extracellular space (ECS). `ffian.flow_model` takes transmembrane- and
compartmental fluid flow into account and predicts the evolution in time
and distribution in space of the volume fractions, 
ion concentrations (Na<sup>+</sup>, K<sup>+</sup>, Cl<sup>-</sup>), 
electrical potentials, and hydrostatic
pressures in the ICS and ECS. In `ffian.zero_flow_model`, 
the transmembrane- and compartmental fluid flow is assumed to be zero 
(corresponding to the model presented in 
[Halnes et al. 2013](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003386)).
The fluid model is presented in Sætra et al. 2023 [XXX].

## Documentation

Documentation at [https://martejulie.github.io/fluid-flow-in-astrocyte-networks](https://martejulie.github.io/fluid-flow-in-astrocyte-networks).

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

### Docker

Version 0.1.0 is available as a docker image at 
[Github Packages](https://github.com/martejulie/fluid-flow-in-astrocyte-networks/pkgs/container/fluid-flow-in-astrocyte-networks)
and can be run using

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared ghcr.io/martejulie/fluid-flow-in-astrocyte-networks:v0.1.0
```

### Source

To install the `ffian`-library from source, navigate to the root of the repository and
run the following commands from the command line:
```bash
python3 -m pip install python/. --upgrade
```

`ffian` requires `fenics-dolfin`, `numpy`, and `matplotlib`.

## Run simulations

The example folder includes code showing how to run the simulations. 
To reproduce the results presented in Sætra et al. 2023, see
[https://github.com/martejulie/fluid-flow-in-astrocyte-networks-analysis](https://github.com/martejulie/fluid-flow-in-astrocyte-networks-analysis).
