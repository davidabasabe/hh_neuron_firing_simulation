import numpy as np
from typing import Callable

from numerical_solvers import calculate_exp_euler_value_matrix_form
from numerical_solvers import extract_A_and_g, calculate_exp_euler_value


def calculate_alpha_constant_m(volt: float) -> float:
    """
    Calculates the alpha rate constant for the m gating variable.

    Inputs:
        volt:   membrane potential in V

    Outputs:
        alpha_m:    alpha rate constant for m
    """
    numerator = 2.5 - (100 * volt)
    denominator = np.exp(numerator) - 1
    return 1000 * numerator / denominator

def calculate_alpha_constant_n(volt: float) -> float:
    """
    Calculates the alpha rate constant for the n gating variable.

    Inputs:
        volt:   membrane potential in V

    Outputs:
        alpha_n:    alpha rate constant for n
    """
    numerator = 0.1 - (10 * volt)
    denominator = np.exp(1 - (100 * volt)) - 1
    return 1000 * numerator / denominator

def calculate_alpha_constant_h(volt: float) -> float:
    """
    Calculates the alpha rate constant for the h gating variable.

    Inputs:
        volt:   membrane potential in V

    Outputs:
        alpha_h:    alpha rate constant for h
    """
    return 70 * np.exp(-50 * volt)

def calculate_alpha_constant(channel: str, volt: float) -> float:
    """
    Dispatches the alpha rate constant calculation for the given channel.

    Inputs:
        channel:    gating variable channel ('m', 'n', or 'h')
        volt:       membrane potential in V

    Outputs:
        alpha:      alpha rate constant for the specified channel
    """
    if channel == 'm':
        return calculate_alpha_constant_m(volt)
    elif channel == 'n':
        return calculate_alpha_constant_n(volt)
    elif channel == 'h':
        return calculate_alpha_constant_h(volt)
    else:
        raise ValueError('Invalid channel')

def calculate_beta_constant_m(volt: float) -> float:
    """
    Calculates the beta rate constant for the m gating variable.

    Inputs:
        volt:   membrane potential in V

    Outputs:
        beta_m:     beta rate constant for m
    """
    return 4000 * np.exp(-500 * volt / 9)

def calculate_beta_constant_n(volt: float) -> float:
    """
    Calculates the beta rate constant for the n gating variable.

    Inputs:
        volt:   membrane potential in V

    Outputs:
        beta_n:     beta rate constant for n
    """
    return 125 * np.exp(-25 * volt / 2)

def calculate_beta_constant_h(volt: float) -> float:
    """
    Calculates the beta rate constant for the h gating variable.

    Inputs:
        volt:   membrane potential in V

    Outputs:
        beta_h:     beta rate constant for h
    """
    denominator = np.exp(3 - (100 * volt)) + 1
    return 1000 / denominator

def calculate_beta_constant(channel: str, volt: float) -> float:
    """
    Dispatches the beta rate constant calculation for the given channel.

    Inputs:
        channel:    gating variable channel ('m', 'n', or 'h')
        volt:       membrane potential in V

    Outputs:
        beta:       beta rate constant for the specified channel
    """
    if channel == 'm':
        return calculate_beta_constant_m(volt)
    elif channel == 'n':
        return calculate_beta_constant_n(volt)
    elif channel == 'h':
        return calculate_beta_constant_h(volt)
    else:
        raise ValueError('Invalid channel')

def calculate_k_constant(temperature: float) -> float:
    """
    Calculates the temperature scaling factor k (Q10 correction).

    Inputs:
        temperature:    simulation temperature in °C

    Outputs:
        k:              temperature scaling factor
    """
    return 3 ** (0.1 * (temperature - 6.3))

def calculate_all_constants(volt: float, temperature: float) -> np.ndarray:
    """
    Calculates all alpha, beta, and k constants for a single voltage value.

    Inputs:
        volt:           membrane potential in V
        temperature:    simulation temperature in °C

    Outputs:
        constants:      array of [alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n, k]
    """

    alpha_m = calculate_alpha_constant("m", volt)
    beta_m = calculate_beta_constant("m", volt)

    alpha_h = calculate_alpha_constant("h", volt)
    beta_h = calculate_beta_constant("h", volt)

    alpha_n = calculate_alpha_constant("n", volt)
    beta_n = calculate_beta_constant("n", volt)

    k = calculate_k_constant(temperature)

    return np.array([alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n, k])

def calculate_all_constants_array(volt_array: np.ndarray, temperature: float) -> np.ndarray:
    """
    Calculates all alpha, beta, and k constants for an array of voltage values.

    Inputs:
        volt_array:     array of membrane potentials in V
        temperature:    simulation temperature in °C

    Outputs:
        constants_array:    matrix of constants (len(volt_array) x 7),
                            columns being [alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n, k]
    """

    constants_array = np.zeros((len(volt_array), 7))
    for index in range(len(volt_array)):
        constants_array[index, :] = calculate_all_constants(volt_array[index], temperature)

    return constants_array

def calculate_steady_state_value(alpha: float, beta: float) -> float:
    """
    Calculates the steady-state value of a gating variable.

    Inputs:
        alpha:  alpha rate constant
        beta:   beta rate constant

    Outputs:
        x_inf:  steady-state gating variable value
    """
    return alpha / (alpha + beta)

def calculate_time_constant_value(alpha: float, beta: float, k: float) -> float:
    """
    Calculates the time constant of a gating variable.

    Inputs:
        alpha:  alpha rate constant
        beta:   beta rate constant
        k:      temperature scaling factor

    Outputs:
        tau:    time constant in s
    """
    return 1 / (k * (alpha + beta))

def calculate_steady_state_over_voltages(volt_limits: (float, float),
                                         channel: str,
                                         volt_step: float = None) -> (np.array, np.array):
    """
    Calculates the steady-state gating variable value over a range of voltages.

    Inputs:
        volt_limits:    tuple of (volt_min, volt_max) in V
        channel:        gating variable channel ('m', 'n', or 'h')
        volt_step:      voltage step size in V (default: volt_range / 100)

    Outputs:
        volt_values:            array of voltage values in V
        steady_state_values:    array of steady-state gating variable values
    """

    volt_min = volt_limits[0]
    volt_max = volt_limits[1]
    volt_range = volt_max - volt_min
    if volt_step is None:
        volt_step = volt_range / 100

    volt_values = np.arange(volt_min, volt_max + volt_step, volt_step)
    steady_state_values = []
    for volt in volt_values:
        alpha = calculate_alpha_constant(channel, volt)
        beta = calculate_beta_constant(channel, volt)
        temp_steady_state_value = calculate_steady_state_value(alpha, beta)
        steady_state_values.append(temp_steady_state_value)

    steady_state_values = np.array(steady_state_values)
    return volt_values, steady_state_values

def calculate_steady_states_over_voltage_array(volt_array: np.ndarray) -> np.ndarray:
    """
    Calculates the steady-state values for all gating variables over an array of voltages.

    Inputs:
        volt_array:     array of membrane potentials in V

    Outputs:
        steady_state_values:    matrix of steady-state values (len(volt_array) x 3),
                                columns being m - n - h
    """

    steady_state_values = np.zeros((len(volt_array), 3))
    channels = ['m', 'n', 'h']
    volt_index = 0
    for volt in volt_array:
        channel_index = 0
        for channel in channels:
            alpha = calculate_alpha_constant(channel, volt)
            beta = calculate_beta_constant(channel, volt)
            temp_steady_state_value = calculate_steady_state_value(alpha, beta)
            steady_state_values[volt_index, channel_index] = temp_steady_state_value
            channel_index += 1
        volt_index += 1

    steady_state_values = np.array(steady_state_values)
    return steady_state_values


def calculate_time_constant_over_voltages(volt_limits: (float, float),
                                         temperature: float,
                                         channel: str,
                                         volt_step: float = None) -> (np.array, np.array):
    """
    Calculates the time constant of a gating variable over a range of voltages.

    Inputs:
        volt_limits:    tuple of (volt_min, volt_max) in V
        temperature:    simulation temperature in °C
        channel:        gating variable channel ('m', 'n', or 'h')
        volt_step:      voltage step size in V (default: volt_range / 100)

    Outputs:
        volt_values:            array of voltage values in V
        time_constant_values:   array of time constant values in s
    """
    volt_min = volt_limits[0]
    volt_max = volt_limits[1]
    volt_range = volt_max - volt_min
    k = calculate_k_constant(temperature)
    if volt_step is None:
        volt_step = volt_range / 100

    volt_values = np.arange(volt_min, volt_max + volt_step, volt_step)
    time_constant_values = []
    for volt in volt_values:
        alpha = calculate_alpha_constant(channel, volt)
        beta = calculate_beta_constant(channel, volt)
        temp_time_constant_value = calculate_time_constant_value(alpha, beta, k)
        time_constant_values.append(temp_time_constant_value)

    time_constant_values = np.array(time_constant_values)
    return volt_values, time_constant_values

def create_sodium_current_equation(
        g_sodium: float,
        volt_sodium: float) -> Callable:
    """
    Creates the sodium current equation as a callable.

    Inputs:
        g_sodium:       sodium conductance in S/m²
        volt_sodium:    sodium reversal potential in V

    Outputs:
        equation:       callable of the form f(m, h, voltage) -> I_Na
    """
    return lambda m, h, voltage: g_sodium * (m ** 3) * h * (voltage - volt_sodium)

def create_potassium_current_equation(
        g_potassium: float,
        volt_potassium: float) -> Callable:
    """
    Creates the potassium current equation as a callable.

    Inputs:
        g_potassium:    potassium conductance in S/m²
        volt_potassium: potassium reversal potential in V

    Outputs:
        equation:       callable of the form f(n, voltage) -> I_K
    """
    return lambda n, voltage: g_potassium * (n ** 4) * (voltage - volt_potassium)

def create_leak_current_equation(
        g_leak: float,
        volt_leak: float) -> Callable:
    """
    Creates the leak current equation as a callable.

    Inputs:
        g_leak:     leak conductance in S/m²
        volt_leak:  leak reversal potential in V

    Outputs:
        equation:   callable of the form f(voltage) -> I_L
    """
    return lambda voltage: g_leak * (voltage - volt_leak)

def create_gate_equation(alpha: float, beta: float, k: float) -> Callable:
    """
    Creates the gating variable ODE as a callable.

    Inputs:
        alpha:  alpha rate constant
        beta:   beta rate constant
        k:      temperature scaling factor

    Outputs:
        equation:   callable of the form f(t, x) -> dx/dt
    """
    return lambda t, actual_gate_value: (alpha * (1 - actual_gate_value) - beta * actual_gate_value) * k


def generate_stair_signal(amplitude_array: list,
                          time_limits: (float, float),
                          time_step: float,
                          pulse_duration: float,
                          gap_duration: float) -> (np.array, np.array):
    """
    Generates a staircase pulse signal with alternating pulses and gaps.

    Inputs:
        amplitude_array:    list of amplitude values for each successive pulse
        time_limits:        tuple of (t_start, t_end) in s
        time_step:          time step in s
        pulse_duration:     duration of each pulse in s
        gap_duration:       duration of each gap between pulses in s

    Outputs:
        time_array:         array of time values in s
        signal_values:      array of signal amplitude values
    """

    time_array = np.arange(time_limits[0], time_limits[1] + time_step, time_step)
    signal_values = []
    amplitude_index = 0
    current_pulse_duration = 0
    current_gap_duration = 0
    is_pulse = True

    for time_value in time_array:
        if is_pulse and amplitude_index < len(amplitude_array):
            signal_values.append(amplitude_array[amplitude_index])
            current_pulse_duration += time_step
        else:
            signal_values.append(0)
            current_gap_duration += time_step

        if current_pulse_duration >= pulse_duration:
            is_pulse = False
            current_pulse_duration = 0
            amplitude_index += 1

        elif current_gap_duration >= gap_duration:
            is_pulse = True
            current_gap_duration = 0

    signal_values = np.array(signal_values)

    return time_array, signal_values

def calculate_axonal_resistance(rho: float, length: float, radius: float) -> float:
    """
    Calculates the axonal resistance of a cylindrical compartment.

    Inputs:
        rho:    axonal resistivity in Ohm*m
        length: compartment length in m
        radius: axon radius in m

    Outputs:
        R_a:    axonal resistance in Ohm
    """
    return (rho * length) / ((radius **2) * np.pi)

def calculate_next_gating_variables(all_constants_array: np.ndarray, dt, curr_gate: np.ndarray) -> np.ndarray:
    """
    Calculates the next gating variables using exponential Euler integration (scalar form).

    Inputs:
        all_constants_array:    array of [alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n, k]
        dt:                     time step in s
        curr_gate:              current gating variables [m, n, h]

    Outputs:
        next_gate:              updated gating variables [m, n, h] at the next time step
    """
    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n, k = all_constants_array
    m, n, h = curr_gate

    # build and solve gating ODEs individually using exponential Euler
    gating_equation_m = create_gate_equation(alpha_m, beta_m, k)
    gating_equation_h = create_gate_equation(alpha_h, beta_h, k)
    gating_equation_n = create_gate_equation(alpha_n, beta_n, k)

    A_m, g_func_m = extract_A_and_g(gating_equation_m)
    next_m = calculate_exp_euler_value(g_func_m, A_m, dt, m, 0)

    A_h, g_func_h = extract_A_and_g(gating_equation_h)
    next_h = calculate_exp_euler_value(g_func_h, A_h, dt, h, 0)

    A_n, g_func_n = extract_A_and_g(gating_equation_n)
    next_n = calculate_exp_euler_value(g_func_n, A_n, dt, n, 0)

    return np.array([next_m, next_n, next_h])


def create_C_matrix(n):
    """
    Creates the tridiagonal compartment coupling matrix C of size n x n.

    The matrix represents second-order spatial finite differences along the axon,
    with boundary conditions applied at the first and last compartments.

    Inputs:
        n:  number of compartments

    Outputs:
        C:  tridiagonal coupling matrix (n x n)
    """

    # Main diagonal
    main_diag = -2 * np.ones(n)
    main_diag[0] = -1
    main_diag[-1] = -1

    # Off diagonals (sub and super diagonals)
    off_diag = np.ones(n - 1)

    # Create the tridiagonal matrix
    C = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

    return C

def calculate_ionic_currents_array(volt_array: np.ndarray,
                                   gate_variables: np.ndarray,
                                   ionic_curr_eqs: np.ndarray) -> np.ndarray:
    """
    Calculates ionic currents for all compartments at a given time step.

    Inputs:
        volt_array:     array of membrane potentials in V (len(compartments))
        gate_variables: matrix of gating variables (len(compartments) x 3), columns being m - n - h
        ionic_curr_eqs: array of ionic current callables [sodium_eq, potassium_eq, leak_eq]

    Outputs:
        ionic_currents: matrix of ionic currents (len(compartments) x 3), columns being I_Na - I_K - I_L
    """

    ionic_currents = np.zeros((len(volt_array), 3))
    sodium_current_eq, potassium_current_eq, leak_current_eq = ionic_curr_eqs

    for index in range(len(volt_array)):
        temp_m, temp_n, temp_h = gate_variables[index]
        temp_v = volt_array[index]
        new_sod_current = sodium_current_eq(temp_m, temp_h, temp_v)
        new_pot_current = potassium_current_eq(temp_n, temp_v)
        new_leak_current = leak_current_eq(temp_v)
        temp_ion_currents = np.array([new_sod_current, new_pot_current, new_leak_current])
        ionic_currents[index] = temp_ion_currents

    return ionic_currents


def compute_pcolormesh_edges(array):
    """
    Computes bin edges for use with pcolormesh from a array of centre values.

    Inputs:
        array:  1D array of centre values

    Outputs:
        edges:  1D array of bin edges, length len(array) + 1
    """
    diffs = np.diff(array) / 2
    edges = np.zeros(len(array) + 1)
    edges[1:-1] = array[:-1] + diffs
    edges[0] = array[0] - diffs[0]
    edges[-1] = array[-1] + diffs[-1]
    return edges

def calculate_next_gating_variables_mat(all_constants_array: np.ndarray, dt, curr_gate: np.ndarray) -> np.ndarray:
    """
    Calculates the next gating variables using exponential Euler integration (matrix form).

    Inputs:
        all_constants_array:    array of [alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n, k]
        dt:                     time step in s
        curr_gate:              current gating variables [m, n, h]

    Outputs:
        next_gating_variables:  updated gating variables [m, n, h] at the next time step
    """

    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n, k = all_constants_array

    # build diagonal entries of the A matrix (one per gating variable)
    first_diag = -k * (alpha_m + beta_m)
    second_diag = -k * (alpha_n + beta_n)
    third_diag = -k * (alpha_h + beta_h)

    diag_vals = np.array([first_diag, second_diag, third_diag])
    a_mat = np.diag(diag_vals)

    # build the b vector (steady-state driving terms)
    b_first_element = alpha_m * k
    b_second_element = alpha_n * k
    b_third_element = alpha_h * k

    b_vector = np.array([b_first_element, b_second_element, b_third_element])

    next_gating_variables = calculate_exp_euler_value_matrix_form(a_mat, b_vector, curr_gate, dt)

    return next_gating_variables

def calculate_potential_field_value(resistivity: float, current: float, distance_x: float, distance_y: float) -> float:
    """
    Calculates the extracellular potential at a point due to a monopolar current source.

    Inputs:
        resistivity:    medium resistivity in Ohm*m
        current:        source current in A
        distance_x:     axial distance from source to point in m
        distance_y:     radial distance from source to point in m

    Outputs:
        potential:      extracellular potential at the point in V
    """
    total_distance = np.sqrt(distance_x ** 2 + distance_y ** 2)
    return resistivity * current / (4 * np.pi * total_distance)

def calculate_potentials_over_distance(resistivity: float,
                                       current: float,
                                       axon_length: float,
                                       length_step: float,
                                       distance_to_axon: float) -> (np.ndarray, np.ndarray):

    # Current source is positioned central to axon
    lower_boundary = -axon_length / 2
    upper_boundary = -lower_boundary

    x_distances = np.arange(lower_boundary, upper_boundary, length_step)

    # Initialise potentials vector
    potentials = np.zeros(len(x_distances))

    # Calculate each potential iteratively
    for index in range(len(x_distances)):
        potentials[index] = calculate_potential_field_value(resistivity, current, x_distances[index], distance_to_axon)

    return potentials, np.arange(0, axon_length, length_step)

def calculate_electric_field_value(resistivity: float,
                                   current: float,
                                   distance_x: float,
                                   distance_y: float) -> float:
    """
    Calculates the axial component of the electric field at a point due to a monopolar current source.

    Inputs:
        resistivity:    medium resistivity in Ohm*m
        current:        source current in A
        distance_x:     axial distance from source to point in m
        distance_y:     radial distance from source to point in m

    Outputs:
        electric_field_value:   axial electric field at the point in V/m
    """
    denom_value = np.sqrt((distance_x ** 2 + distance_y ** 2) ** 3)
    electric_field_value = distance_x * resistivity * current / (denom_value * 4 * np.pi)
    return electric_field_value

def calculate_electric_field_over_distance(resistivity: float,
                                           current: float,
                                           axon_length: float,
                                           length_step: float,
                                           distance_to_axon: float) -> (np.ndarray, np.ndarray):
    """
    Calculates the axial electric field along the axon due to a centrally placed monopolar current source.

    Inputs:
        resistivity:        medium resistivity in Ohm*m
        current:            source current in A
        axon_length:        total axon length in m
        length_step:        spatial step size in m
        distance_to_axon:   radial distance from the current source to the axon in m

    Outputs:
        electric_field:     array of axial electric field values in V/m
        positions:          array of axon positions in m (starting from 0)
    """
    # Current source is positioned central to axon
    lower_boundary = -axon_length / 2
    upper_boundary = -lower_boundary

    x_distances = np.arange(lower_boundary, upper_boundary, length_step)
    electric_field = np.zeros(len(x_distances))

    for index in range(len(x_distances)):
        electric_field[index] = calculate_electric_field_value(resistivity, current, x_distances[index], distance_to_axon)

    return electric_field, np.arange(0, axon_length, length_step)

def calculate_activating_function_value(resistivity: float,
                                        current: float,
                                        distance_x: float,
                                        distance_y: float) -> float:
    """
    Calculates the activating function value at a point due to a monopolar current source.

    The activating function is the second spatial derivative of the extracellular potential
    along the axon axis, and serves as a predictor of membrane excitation.

    Inputs:
        resistivity:    medium resistivity in Ohm*m
        current:        source current in A
        distance_x:     axial distance from source to point in m
        distance_y:     radial distance from source to point in m

    Outputs:
        activation_value:   activating function value at the point in V/m²
    """
    coefficient = resistivity * current / (4 * np.pi)
    first_derivative_term = 3 * (distance_x ** 2) / (np.sqrt((distance_x ** 2 + distance_y ** 2) ** 5))
    second_derivative_term = -1 / (np.sqrt((distance_x ** 2 + distance_y ** 2) ** 3))
    activation_value = coefficient * (first_derivative_term + second_derivative_term)
    return activation_value

def calculate_activating_function_values_over_distance(resistivity: float,
                                                       current: float,
                                                       axon_length: float,
                                                       length_step: float,
                                                       distance_to_axon: float) -> (np.ndarray, np.ndarray):
    """
    Calculates the activating function along the axon due to a centrally placed monopolar current source.

    Inputs:
        resistivity:        medium resistivity in Ohm*m
        current:            source current in A
        axon_length:        total axon length in m
        length_step:        spatial step size in m
        distance_to_axon:   radial distance from the current source to the axon in m

    Outputs:
        activating_values:  array of activating function values in V/m²
        positions:          array of axon positions in m (starting from 0)
    """
    # Current source is positioned central to axon
    lower_boundary = -axon_length / 2
    upper_boundary = -lower_boundary

    x_distances = np.arange(lower_boundary, upper_boundary, length_step)
    activating_values = np.zeros(len(x_distances))

    for index in range(len(x_distances)):
        activating_values[index] = calculate_activating_function_value(resistivity, current, x_distances[index], distance_to_axon)

    return activating_values, np.arange(0, axon_length, length_step)