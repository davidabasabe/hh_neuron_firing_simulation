# Check this website out for forward euler https://math.libretexts.org/Bookshelves/Differential_Equations/Numerically_Solving_Ordinary_Differential_Equations_(Brorson)/01%3A_Chapters/1.02%3A_Forward_Euler_method
# These ones for Heun's Method https://testbook.com/maths/heuns-method https://en.wikipedia.org/wiki/Heun%27s_method
# For exp. euler: https://www.cambridge.org/core/services/aop-cambridge-core/content/view/8ED12FD70C2491C4F3FB7A0ACF922FCD/S0962492910000048a.pdf/exponential-integrators.pdf

from typing import Callable, Any
import numpy as np
import sympy as sp
from scipy.linalg import expm

def calculate_euler_next_value(rhs_function: Callable,
                               current_val: float,
                               current_time: float,
                               time_step: float) -> float:
    """
    Calculates the next value using a single forward Euler step.

    Inputs:
        rhs_function:   right-hand side function f(t, y)
        current_val:    current value y_n
        current_time:   current time t_n in s
        time_step:      time step h in s

    Outputs:
        next_val:       next value y_n+1
    """
    current_rhs_value = rhs_function(current_time, current_val)
    return current_val + (time_step * current_rhs_value)


def forward_explicit_euler_solver(rhs_function: Callable,
                                  time_step: float,
                                  time_limits: (float, float) = (0, 10),
                                  initial_condition_value: float = 0) -> (np.ndarray, np.ndarray):
    """
    Solves an ODE using the forward explicit Euler method.

    Inputs:
        rhs_function:               right-hand side function f(t, y)
        time_step:                  time step h in s
        time_limits:                tuple of (t_start, t_end) in s (default: (0, 10))
        initial_condition_value:    initial value y(t_start) (default: 0)

    Outputs:
        time_array:             array of time values in s
        approx_solution_values: array of approximate solution values
    """

    # dy/dt = f(t,y) -> dy/dt approximately is (y_n+1 + y_n) / h, h being the time step between y values after discretising
    # Solving for y_n+1 -> y_n+1 = y_n + h * f(t,y) -> next value is function of current value, time step, and rhs function
    # Do this for every value

    start_time = time_limits[0]
    end_time = time_limits[1]
    input_time_step = time_step

    time_array = [start_time]
    approx_values_array = [initial_condition_value]
    value_index = 0

    while time_array[-1] + input_time_step < end_time:
        current_value = approx_values_array[value_index]
        current_time = time_array[value_index]
        next_value = calculate_euler_next_value(rhs_function, current_value, current_time, input_time_step)
        approx_values_array.append(next_value)

        next_time = current_time + input_time_step
        time_array.append(next_time)

        value_index += 1

    time_array = np.array(time_array)
    approx_solution_values = np.array(approx_values_array)

    return time_array, approx_solution_values

def heun_second_order_solver(rhs_function: Callable,
                                  time_step: float,
                                  time_limits: (float, float) = (0, 10),
                                  initial_condition_value: float = 0) -> (np.ndarray, np.ndarray):
    """
    Solves an ODE using Heun's second-order predictor-corrector method.

    Inputs:
        rhs_function:               right-hand side function f(t, y)
        time_step:                  time step h in s
        time_limits:                tuple of (t_start, t_end) in s (default: (0, 10))
        initial_condition_value:    initial value y(t_start) (default: 0)

    Outputs:
        time_array:             array of time values in s
        approx_solution_values: array of approximate solution values
    """

    # Heun's method first calculates a value using Euler's method, but then calculates a new intermediate
    # value by using the original function again, now with a move forward in time t_n + h, and using the euler's
    # method calculated value as the new current value. An intermediate value between the new value and the current one
    # is then calculated

    start_time = time_limits[0]
    end_time = time_limits[1]
    time_array = [start_time]
    value_index = 0
    approx_values_array = [initial_condition_value]

    while time_array[-1] + time_step < end_time:
        current_value = approx_values_array[value_index]
        current_time = time_array[value_index]
        current_rhs_result = rhs_function(current_time, current_value)

        euler_value = calculate_euler_next_value(rhs_function, current_value, current_time, time_step)
        updated_time = time_array[value_index] + time_step
        updated_value = rhs_function(updated_time, euler_value)

        intermediate_value = (current_rhs_result + updated_value) * time_step / 2
        approx_sol_value = current_value + intermediate_value
        approx_values_array.append(approx_sol_value)

        next_time = current_time + time_step
        time_array.append(next_time)

        value_index += 1

    time_array = np.array(time_array)
    approx_solution_values = np.array(approx_values_array)

    return time_array, approx_solution_values

def extract_A_and_g(diff_equation: Callable):
    """
    Extracts the linear coefficient A and the remaining term g(t) from a linear ODE of the form
    dy/dt = A*y + g(t), using symbolic differentiation.

    Inputs:
        diff_equation:  callable of the form f(t, y) representing the ODE right-hand side

    Outputs:
        A:      linear coefficient (float)
        g_func: callable g(t), the remaining non-linear term after removing the A*y part
    """
    t, v = sp.symbols('t v')
    symbolic_diff_eq = diff_equation(t, v)
    A = sp.diff(symbolic_diff_eq, v)
    A = float(A)
    g_expr = symbolic_diff_eq.subs(v, 0)
    g_func = sp.lambdify(t, g_expr)
    return A, g_func

def phi1(z):
    """
    Evaluates the phi_1 function used in exponential integrators: phi_1(z) = (exp(z) - 1) / z.
    Returns 1 in the limit as z -> 0 to avoid division by zero.

    Inputs:
        z:  scalar argument

    Outputs:
        phi_1(z):   scalar result
    """
    return (np.exp(z) - 1) / z if z != 0 else 1

def calculate_exp_euler_value(g_func: Callable,
                              A_matrix: Any,
                              time_step: float,
                              current_value: float,
                              current_time: float) -> float:
    """
    Calculates the next value using the scalar exponential Euler method.

    Inputs:
        g_func:         callable g(t), the non-linear part of the ODE
        A_matrix:       linear coefficient A (scalar or matrix)
        time_step:      time step h in s
        current_value:  current value y_n
        current_time:   current time t_n in s

    Outputs:
        exp_euler_value:    next value y_n+1
    """

    time_step_times_A = time_step * A_matrix
    exp_val = np.exp(time_step_times_A) * current_value
    right_value = time_step * phi1(time_step_times_A) * g_func(current_time)
    exp_euler_value = right_value + exp_val

    return exp_euler_value

def calculate_exp_euler_value_matrix_form(a_matrix: np.ndarray,
                                          b_vector: np.ndarray,
                                          current_values: np.ndarray,
                                          dt: float) -> np.ndarray:
    """
    Calculates the next values using the matrix exponential Euler method.

    Solves: y_n+1 = exp(A*h) * y_n + phi_1(A*h) * b

    Inputs:
        a_matrix:       linear coefficient matrix A (n x n)
        b_vector:       constant driving vector b (n,)
        current_values: current state vector y_n (n,)
        dt:             time step h in s

    Outputs:
        next_values:    next state vector y_n+1 (n,)
    """
    exp_a = expm(a_matrix * dt)
    phi1 = np.linalg.solve(a_matrix, exp_a - np.eye(3))

    next_values = (exp_a @ current_values) + (phi1 @ b_vector)
    return  next_values


def euler_exponential_first_order_solver(rhs_function: Callable,
                                         time_step: float,
                                         time_limits: (float, float) = (0, 10),
                                         initial_condition_value: float = 0) -> (np.ndarray, np.ndarray):
    """
    Solves an ODE using the exponential Euler method.

    Inputs:
        rhs_function:               right-hand side function f(t, y)
        time_step:                  time step h in s
        time_limits:                tuple of (t_start, t_end) in s (default: (0, 10))
        initial_condition_value:    initial value y(t_start) (default: 0)

    Outputs:
        time_array:             array of time values in s
        approx_solution_values: array of approximate solution values
    """
    start_time = time_limits[0]
    end_time = time_limits[1]
    time_array = [start_time]
    value_index = 0
    approx_values_array = [initial_condition_value]

    A, g_func = extract_A_and_g(rhs_function)

    while time_array[-1] + time_step < end_time:
        current_value = approx_values_array[value_index]
        current_time = time_array[value_index]
        next_value = calculate_exp_euler_value(g_func, phi1, A, time_step, current_value, current_time)
        approx_values_array.append(next_value)

        next_time = current_time + time_step
        time_array.append(next_time)

        value_index += 1

    time_array = np.array(time_array)
    approx_solution_values = np.array(approx_values_array)

    return time_array, approx_solution_values