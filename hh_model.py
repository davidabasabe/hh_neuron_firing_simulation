import numpy as np
import time
from hh_utils import (calculate_all_constants_array,
                      create_potassium_current_equation,
                      create_sodium_current_equation,
                      create_leak_current_equation,
                      calculate_steady_states_over_voltage_array,
                      create_C_matrix, calculate_ionic_currents_array, calculate_axonal_resistance,
                      calculate_next_gating_variables_mat)


def hh_gating(V: np.ndarray, dt: float, curr_gate: np.ndarray, T: float) -> np.ndarray:
    """
    Calculates the gating variables for a future time step.

    Inputs:
        V:          Membrane potential of the current time step in V
        dt:         time step in s
        curr_gate:  gating variables of the current time step (3x1 vector)
        T:          Simulation temperature in °C

    Outputs:
        new_gate:       gating variables of the next time step (vector 3x1)
    """

## 1) calculate gating variables

    all_constants_array = calculate_all_constants_array(volt_array=V, temperature=T)
    volt_array_len = len(V)
    next_gate_var_array = np.zeros((volt_array_len, 3))

    # iterate over each compartment and compute next gating variables
    for index in range(volt_array_len):
        compartment_gates = curr_gate[index]
        compartment_constants = all_constants_array[index]
        next_gate_var_array[index] = calculate_next_gating_variables_mat(compartment_constants, dt, compartment_gates)

    return next_gate_var_array


def hh_potential(V: np.ndarray,
                 dt: float,
                 I_ions: np.ndarray,
                 V_ext: np.ndarray,
                 capacitance_m: float,
                 c_matrix: np.ndarray,
                 axon_resistance: float) -> np.ndarray:
    """
    Calculates the membrane potential for a future time step.

    Inputs:
        V:              Membrane potential of the current time step in V
        dt:             time step in s
        I_ions:         ionic currents of a current timestep (3x1 vector)
        V_ext:          external potential vector for the current timestep
        capacitance_m:  membrane capacitance in F
        c_matrix:       compartment coupling matrix
        axon_resistance: axonal resistance in Ohm

    Outputs:
        V_new:      Membrane potential of a future timestep
    """

    ## 1) calculate new membrane potential
    # parameters

    volt_len = len(V)
    # build the implicit system matrix A = I - dt * C / (Cm * Ra)
    a_matrix = np.eye(volt_len) - (dt * c_matrix / (capacitance_m * axon_resistance))
    # sum ionic currents across all ion types for each compartment
    I_ions_total = np.sum(I_ions, axis=1)
    # build the right-hand side vector incorporating ionic and external contributions
    b_vector = V + ((dt / capacitance_m) * (-I_ions_total)) + ((dt * c_matrix @ V_ext) / (capacitance_m * axon_resistance))
    # solve the linear system A * V_new = b
    next_voltage_array = np.linalg.solve(a_matrix, b_vector)

    return next_voltage_array


def hh_model(V_ext: np.ndarray,
             t_end: float,
             dt: float,
             T: float,
             g_sodium: float,
             volt_sodium: float,
             g_potassium: float,
             volt_potassium: float,
             g_leak: float,
             volt_leak: float,
             capacitance_m: float,
             axonal_rho: float,
             axonal_length: float,
             axonal_radius: float,
             V_rest: float = 0,
             compartment_size: int = 100,
             c_matrix_mode: str = "full") -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Simulates a multi-compartment Hodgkin-Huxley neuron model.

    Inputs:
        V_ext:              external potential matrix over time (len(t) x compartment_size) in V
        t_end:              simulation duration in s
        dt:                 time step in s
        T:                  simulation temperature in °C
        g_sodium:           sodium conductance in S/m²
        volt_sodium:        sodium reversal potential in V
        g_potassium:        potassium conductance in S/m²
        volt_potassium:     potassium reversal potential in V
        g_leak:             leak conductance in S/m²
        volt_leak:          leak reversal potential in V
        capacitance_m:      membrane capacitance in F
        axonal_rho:         axonal resistivity in Ohm*m
        axonal_length:      length of each compartment in m
        axonal_radius:      radius of the axon in m
        V_rest:             membrane resting potential in V (default: 0)
        compartment_size:   number of compartments (default: 100)
        c_matrix_mode:      coupling mode, "full" for full C matrix or anything else for no coupling (default: "full")

    Outputs:
        V:          membrane potentials as a matrix (len(t) x compartment_size)
        gates:      gating variables as a matrix (len(t) x compartment_size x 3), columns being m - n - h
        I_ions:     ion currents as a matrix (len(t) x compartment_size x 3), columns being i_na - i_k - i_l
        t:          time vector
    """

    ## Definitions and constants

    # build compartment coupling matrix depending on mode
    if c_matrix_mode == "full":
        c_matrix = create_C_matrix(compartment_size)
    else:
        c_matrix = np.zeros((compartment_size, compartment_size))

    axonal_resistance = calculate_axonal_resistance(axonal_rho, axonal_length, axonal_radius)

    # running time
    t = np.arange(0, t_end + dt, dt)

    # Potential vector 1xlength
    V = np.zeros((len(t), compartment_size))

    # all matrices are 3xlength, with the rows always being m - n - h
    gates = np.zeros((len(t), compartment_size, 3))
    I_ions = np.zeros((len(t), compartment_size, 3))

    # Initialise ionic current equations
    sodium_current_eq = create_sodium_current_equation(g_sodium, volt_sodium) # Input: m, h, voltage
    potassium_current_eq = create_potassium_current_equation(g_potassium, volt_potassium) # Input: n, voltage
    leak_current_eq = create_leak_current_equation(g_leak, volt_leak) # Input: voltage

    ionic_curr_eqs = np.array([sodium_current_eq, potassium_current_eq, leak_current_eq])

    ## Initial calculations
    # initialize first voltage value

    V[0, :] = V_rest
    first_volt = V[0, :]

    # calculate the rates (alpha and beta) for initial voltages and
    # use the steady-state equations to obtain the initial gating variable states
    initial_states = calculate_steady_states_over_voltage_array(first_volt)

    gates[0] = initial_states

    # calculate first ionic currents

    I_ions[0] = calculate_ionic_currents_array(first_volt, initial_states, ionic_curr_eqs)

    t_now = time.time()

    ## iterative calculation of the membrane potentials
    for i in range(len(t) - 2):
        f"Time step number: {i + 1} of {int(t_end / dt) - 1} total time steps\n"
        t_loop = time.time()
        actual_v = V[i]
        actual_gate_values = gates[i]
        next_i = i + 1
        next_V_ext = V_ext[next_i]

        # calculate next gating variables of a future timestep
        t_gates = time.time()
        next_gate_values = hh_gating(actual_v, dt, actual_gate_values, T)
        print(f"Gating time duration: {time.time() - t_gates}")

        # calculate ionic currents for a future timestep
        t_ions = time.time()
        next_ionic_currents = calculate_ionic_currents_array(actual_v, next_gate_values, ionic_curr_eqs)
        print(f"Ions time duration: {time.time() - t_ions}")

        t_volts = time.time()
        # calculate membrane potential of a future timestep
        next_voltage = hh_potential(actual_v,
                                    dt,
                                    next_ionic_currents,
                                    next_V_ext,
                                    capacitance_m,
                                    c_matrix,
                                    axonal_resistance)
        print(f"Voltage time duration: {time.time() - t_volts}")

        ## assign outputs
        I_ions[next_i] = next_ionic_currents
        gates[next_i] = next_gate_values
        V[next_i] = next_voltage

        print(f"Duration of this step: {time.time() - t_loop}")


    print(f"Total simulation time: {time.time() - t_now}")
    return V, gates, I_ions, t