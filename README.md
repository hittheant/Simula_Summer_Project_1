# ffian: Fluid Flow In Astrocyte Networks

ffian is an implementation of the KNP continuity equations for a 
one-dimensional system containing two compartments: 
one representing an astrocyte network (ICS) and one representing the
extracellular space (ECS). ffian.flow\_model takes transmembrane- and
compartmental fluid flow into account and predicts the evolution in time
and distribution in space of the volume fractions, 
ion concentrations (Na<sup>+</sup>, K<sup>+</sup>, Cl<sup>-</sup>), 
electrical potentials, and hydrostatic
pressures in the ICS and ECS. In ffian.zero\_flow\_model, 
the transmembrane- and compartmental fluid flow is assumed to be zero 
(corresponding to the model presented in the model presented in 
[Halnes et al. 2013](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003386)).
The model is presented in Sætra et al. 2023 [XXX].

## Installation

## Run simulations

The example folder includes code showing how to run the simulations. 
To reproduce the results presented in Sætra et al. 2023, see
[https://github.com/martejulie/fluid-flow-in-astrocyte-networks-analysis](https://github.com/martejulie/fluid-flow-in-astrocyte-networks-analysis).
