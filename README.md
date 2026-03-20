# hh_neuron_firing_simulation
Repository for an HH Neuron Model Simulation in Python. It utilises the Hodgkin-Huxley Neuron Model, which directly models the gating variables and dynamics of neurons and elicits firing behaviours through external potentials. It is part of a coding exercise from the Neuroprosthetics Course at TUM, for which reason a report is attached. 

## Dependencies
In order to run the code, numpy, sympy, scipy.linalg, and matplotlib are required. The code was written and tested using Python 3.10.

## Code Structure

The code consists of a main script that runs the multiple simulations described in the report, using the model described in hh_model.py and functions in hh_utils for calculating relevant values for the model. Besides that, the script numerical_solvers.py contains implementations of different numerical solvers used to solve the differential equations of the HH Neuron model.

### hh_model.py

This script defines three functions: 
- hh_gating: Calculates the future step gating variables of a neuron using current membrane voltage and gate values and the necessary differential equations
- hh_potential: Calculates the future step membrane voltage based on the current ionic currents, membrane voltage, membrane capacitance, and the axon's resistance
- hh_model: Simulates fully the behaviour of a neuron according to the HH Model over a given period of time by using both hh_gating and hh_potential at each time step to solve the dynamics of the neuron

This script makes use of the modular functions defined in hh_utils.py to simplify calculations and keep hh_model.py as high-level functions, and results from the simulations (resulting voltages and firing behaviours) are automatically saved into .npy files.

### hh_utils.py

This script contains diverse low-level modular functions to perform necessary calculations for all variables in the HH Neuron Model Simualtion. It uses the Exponential Euler numerical method from numerical_solvers.py in order to solve for the next step of a differential equation, which is required to calculate new gating variables and membrane potentials in the simulation. The functions are programmed in a modular way in order to reuse them and build up to the hh_model script.

### numerical_solvers.py

Contains implementations of different numerical solvers, such as Heun's and Euler's, for solving differential equations. The exponential euler method is specifically implemented in a way that allows to solve for matrices, which is especially efficient and fast for the cable equations of an axon.

### svg_simple_plotting.py

This script contains a simple function to plot results from the simulation saved as .npy files and save them into an svg file.
