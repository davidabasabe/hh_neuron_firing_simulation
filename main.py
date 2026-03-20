import numpy as np
from hh_utils import (calculate_steady_state_over_voltages,
                      calculate_time_constant_over_voltages,
                      generate_stair_signal, compute_pcolormesh_edges,
                      calculate_potentials_over_distance,
                      calculate_electric_field_over_distance,
                      calculate_activating_function_values_over_distance)

from hh_model import hh_model
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import matplotlib.ticker as ticker


if __name__ == '__main__':

    ########################
    # External Potentials  #
    ########################

    mpl.rcParams['lines.linewidth'] = 0.5

    # General parameters
    rho_medium = 1
    axon_length = 300e-6
    length_step = 0.1e-6
    distance_to_axon = 10e-6

    # Calculating potentials for the first current
    first_current = 1e-3

    first_current_potentials, distances_array = calculate_potentials_over_distance(rho_medium,
                                                                  first_current,
                                                                  axon_length,
                                                                  length_step,
                                                                  distance_to_axon)

    np.save("data/first_current_potentials.npy", first_current_potentials)

    fig = plt.figure(figsize=(10, 5))
    plt.tick_params(axis='both', labelsize=14)
    #plt.yticks(np.arange(0, 0.011, 0.001))
    plt.plot(distances_array, first_current_potentials)
    plt.title(
        r'External Membrane Potentials for a distance of 10$\mu m$ between current $I_1 = 1mA$ and a $300 \mu m$ long axon')
    #plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000:.1f}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000000:.1f}'))
    plt.ylabel('V(V)', fontsize=14)
    plt.xlabel(r'x($\mu$m)', fontsize=14)
    plt.grid(True)
    plt.savefig("plots/first_current_potentials.svg", format="svg")
    plt.show()

    # Calculating potentials for the second current
    second_current = -1e-3

    second_current_potentials, distances_array = calculate_potentials_over_distance(rho_medium,
                                                                                   second_current,
                                                                                   axon_length,
                                                                                   length_step,
                                                                                   distance_to_axon)

    np.save("data/second_current_potentials.npy", first_current_potentials)

    fig = plt.figure(figsize=(10, 5))
    plt.tick_params(axis='both', labelsize=14)
    # plt.yticks(np.arange(0, 0.011, 0.001))
    plt.plot(distances_array, second_current_potentials)
    plt.title(
        r'External Membrane Potentials for a distance of 10$\mu m$ between current $I_2 = -1mA$ and a $300 \mu m$ long axon')
    # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000:.1f}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000000:.1f}'))
    plt.ylabel('V(V)', fontsize=14)
    plt.xlabel(r'x($\mu$m)', fontsize=14)
    plt.grid(True)
    plt.savefig("plots/second_current_potentials.svg", format="svg")
    plt.show()

    # Calculate electric field for first current

    first_current_e_field, distances_array = calculate_electric_field_over_distance(rho_medium,
                                                                                    first_current,
                                                                                    axon_length,
                                                                                    length_step,
                                                                                    distance_to_axon)
    np.save("data/first_current_e_field.npy", first_current_e_field)

    fig = plt.figure(figsize=(10, 5))
    plt.tick_params(axis='both', labelsize=14)
    plt.plot(distances_array, first_current_e_field)
    plt.title(
        r'External Membrane Electric Field for a distance of 10$\mu m$ between current $I_1 = 1mA$ and a $300 \mu m$ long axon')
    # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000:.1f}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000000:.1f}'))
    plt.ylabel('E(V/m)', fontsize=14)
    plt.xlabel(r'x($\mu$m)', fontsize=14)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.grid(True)
    plt.savefig("plots/first_current_electric_field.svg", format="svg")
    plt.show()

    # Calculate electric field for first current

    second_current_e_field, distances_array = calculate_electric_field_over_distance(rho_medium,
                                                                                    second_current,
                                                                                    axon_length,
                                                                                    length_step,
                                                                                    distance_to_axon)
    np.save("data/second_current_e_field.npy", second_current_e_field)

    fig = plt.figure(figsize=(10, 5))
    plt.tick_params(axis='both', labelsize=14)
    plt.plot(distances_array, second_current_e_field)
    plt.title(
        r'External Membrane Electric Field for a distance of 10$\mu m$ between current $I_2 = -1mA$ and a $300 \mu m$ long axon')
    # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000:.1f}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000000:.1f}'))
    plt.ylabel('E(V/m)', fontsize=14)
    plt.xlabel(r'x($\mu$m)', fontsize=14)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.grid(True)
    plt.savefig("plots/second_current_electric_field.svg", format="svg")
    plt.show()

    # Calculate activation function values for first current

    first_current_activation_values, distances_array = calculate_activating_function_values_over_distance(rho_medium,
                                                                                                  first_current,
                                                                                                  axon_length,
                                                                                                  length_step,
                                                                                                  distance_to_axon)
    np.save("data/first_current_activation_values.npy", first_current_activation_values)

    fig = plt.figure(figsize=(10, 5))
    plt.tick_params(axis='both', labelsize=14)
    plt.plot(distances_array, first_current_activation_values)
    plt.title(
        r'Activation Function for a distance of 10$\mu m$ between current $I_1 = 1mA$ and a $300 \mu m$ long axon')
    # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000:.1f}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000000:.1f}'))
    plt.ylabel(r'A(V/$m^2$)', fontsize=14)
    plt.xlabel(r'x($\mu$m)', fontsize=14)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.grid(True)
    plt.savefig("plots/first_current_activation_values.svg", format="svg")
    plt.show()

    # Calculate activation function values for second current

    second_current_activation_values, distances_array = calculate_activating_function_values_over_distance(rho_medium,
                                                                                                          second_current,
                                                                                                          axon_length,
                                                                                                          length_step,
                                                                                                          distance_to_axon)
    np.save("data/second_current_activation_values.npy", second_current_activation_values)

    fig = plt.figure(figsize=(10, 5))
    plt.tick_params(axis='both', labelsize=14)
    plt.plot(distances_array, second_current_activation_values)
    plt.title(
        r'Activation Function for a distance of 10$\mu m$ between current $I_2 = -1mA$ and a $300 \mu m$ long axon')
    # plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000:.1f}'))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000000:.1f}'))
    plt.ylabel(r'A(V/$m^2$)', fontsize=14)
    plt.xlabel(r'x($\mu$m)', fontsize=14)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.grid(True)
    plt.savefig("plots/second_current_activation_values.svg", format="svg")
    plt.show()

    ########################
    # HH-Model simulations #
    ########################

    # Time Step and Temperature definition
    dt = 25e-6
    temp = 6.3
    t_end = 30e-3

    ## Conductivities definition
    g_sodium = 120e-3
    g_potassium = 36e-3
    g_leak = 0.3e-3

    ## Potentials definition
    volt_sodium = 115e-3
    volt_potassium = -12e-3
    volt_leak = 10.6e-3
    volt_rest = 0

    ## Other constants
    capacitance_membrane = 1e-6
    rho_axon = 7e-3
    radius_axon = 1e-6
    compartment_length = 3e-6 # Acts as length step for calculating activation values
    num_compartments = math.ceil(axon_length / compartment_length)

    # First external potentials
    total_time_steps = t_end / dt
    V_ext_one = np.zeros((int(total_time_steps), num_compartments))

    ## Pulse starting at 5ms with an amplitude of -0.05 mA and 1ms long, mono-phasic
    start_time_pulse = int((5e-3 / dt) - 1)
    end_time_pulse = int(6e-3 / dt)
    first_stim_current = -0.05e-3

    ## Calculate and store external potentials
    for index in range(start_time_pulse, end_time_pulse):
        V_ext_one[index], _ = calculate_potentials_over_distance(rho_medium,
                                                                 first_stim_current,
                                                                 axon_length,
                                                                 compartment_length,
                                                                 distance_to_axon)

    ## Simulate neuron using external potentials

    first_hh_volt_array, first_hh_gates_array, first_hh_ion_currents_array, first_hh_time_array = hh_model(
        V_ext_one,
        t_end,
        dt,
        temp,
        g_sodium,
        volt_sodium,
        g_potassium,
        volt_potassium,
        g_leak,
        volt_leak,
        capacitance_membrane,
        rho_axon,
        compartment_length,
        radius_axon,
        V_rest=volt_rest,
        compartment_size=num_compartments,
        c_matrix_mode="full"
    )

    np.save("data/first_hh_volt_array.npy", first_hh_volt_array)
    np.save("data/first_hh_time_array.npy", first_hh_time_array)

    # Plot using pcolormesh
    time_edges = compute_pcolormesh_edges(first_hh_time_array)
    compartments = np.arange(num_compartments)
    compartment_edges = compute_pcolormesh_edges(compartments)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot with rasterization
    mesh = ax.pcolormesh(
        time_edges,
        compartment_edges,
        first_hh_volt_array.T * 1000,
        shading='auto',
        cmap='viridis',
        vmin=-20,
        vmax=100,
        rasterized=True  # This is the key for compression!
    )

    # Axis formatting
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Compartment Nr.', fontsize=14)
    ax.set_title('Propagation of action potentials with a mono-phasic pulse stimulation $I_1 = -0.05mA$ at t = 5ms')

    # Format tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000:.1f}'))
    ax.tick_params(axis='both', labelsize=14)
    ax.invert_yaxis()

    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Voltage (mV)')

    # Save as compressed SVG
    fig.savefig("plots/first_voltages_plot.svg", format="svg", dpi=300)
    plt.show()

    # Second external potentials
    V_ext_two = np.zeros((int(total_time_steps), num_compartments))

    ## Pulse starting at 5ms with an amplitude of -0.1 mA and 1ms long, mono-phasic
    start_time_pulse = int((5e-3 / dt) - 1)
    end_time_pulse = int(6e-3 / dt)
    second_stim_current = -0.1e-3

    ## Calculate and store external potentials
    for index in range(start_time_pulse, end_time_pulse):
        V_ext_two[index], _ = calculate_potentials_over_distance(rho_medium,
                                                                 second_stim_current,
                                                                 axon_length,
                                                                 compartment_length,
                                                                 distance_to_axon)

    ## Simulate neuron using external potentials
    second_hh_volt_array, second_hh_gates_array, second_hh_ion_currents_array, second_hh_time_array = hh_model(
        V_ext_two,
        t_end,
        dt,
        temp,
        g_sodium,
        volt_sodium,
        g_potassium,
        volt_potassium,
        g_leak,
        volt_leak,
        capacitance_membrane,
        rho_axon,
        compartment_length,
        radius_axon,
        V_rest=volt_rest,
        compartment_size=num_compartments,
        c_matrix_mode="full"
    )

    np.save("data/second_hh_volt_array.npy", second_hh_volt_array)
    np.save("data/second_hh_time_array.npy", second_hh_time_array)
    # Plot using pcolormesh
    time_edges = compute_pcolormesh_edges(second_hh_time_array)
    compartments = np.arange(num_compartments)
    compartment_edges = compute_pcolormesh_edges(compartments)
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot with rasterization
    mesh = ax.pcolormesh(
        time_edges,
        compartment_edges,
        second_hh_volt_array.T * 1000,
        shading='auto',
        cmap='viridis',
        vmin=-20,
        vmax=100,
        rasterized=True
    )
    # Axis formatting
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Compartment Nr.', fontsize=14)
    ax.set_title('Propagation of action potentials with a mono-phasic pulse stimulation $I_2 = -0.1mA$ at t = 5ms')
    # Format tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000:.1f}'))
    ax.tick_params(axis='both', labelsize=14)
    ax.invert_yaxis()
    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Voltage (mV)')
    # Save as compressed SVG
    fig.savefig("plots/second_voltages_plot.svg", format="svg", dpi=300)
    plt.show()

    # Third external potentials
    V_ext_three = np.zeros((int(total_time_steps), num_compartments))

    ## Pulse starting at 5ms with an amplitude of -0.1 mA and 1ms long, bi-phasic
    start_time_pulse = int((5e-3 / dt) - 1)
    mid_time_pulse = int(6e-3 / dt)
    end_time_pulse = int(7e-3 / dt)
    third_stim_current = -0.1e-3

    ## Calculate and store external potentials
    for index in range(start_time_pulse, end_time_pulse):
        if index == mid_time_pulse - 1:
            third_stim_current = -third_stim_current
        V_ext_three[index], _ = calculate_potentials_over_distance(rho_medium,
                                                                 third_stim_current,
                                                                 axon_length,
                                                                 compartment_length,
                                                                 distance_to_axon)

    ## Simulate neuron using external potentials
    third_hh_volt_array, third_hh_gates_array, third_hh_ion_currents_array, third_hh_time_array = hh_model(
        V_ext_three,
        t_end,
        dt,
        temp,
        g_sodium,
        volt_sodium,
        g_potassium,
        volt_potassium,
        g_leak,
        volt_leak,
        capacitance_membrane,
        rho_axon,
        compartment_length,
        radius_axon,
        V_rest=volt_rest,
        compartment_size=num_compartments,
        c_matrix_mode="full"
    )

    np.save("data/third_hh_volt_array.npy", third_hh_volt_array)
    np.save("data/third_hh_time_array.npy", third_hh_time_array)
    # Plot using pcolormesh
    time_edges = compute_pcolormesh_edges(third_hh_time_array)
    compartments = np.arange(num_compartments)
    compartment_edges = compute_pcolormesh_edges(compartments)
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot with rasterization
    mesh = ax.pcolormesh(
        time_edges,
        compartment_edges,
        third_hh_volt_array.T * 1000,
        shading='auto',
        cmap='viridis',
        vmin=-20,
        vmax=100,
        rasterized=True
    )
    # Axis formatting
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Compartment Nr.', fontsize=14)
    ax.set_title('Propagation of action potentials with a bi-phasic pulse stimulation $I_3 = -0.1mA$ at t = 5ms')
    # Format tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000:.1f}'))
    ax.tick_params(axis='both', labelsize=14)
    ax.invert_yaxis()
    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Voltage (mV)')
    # Save as compressed SVG
    fig.savefig("plots/third_voltages_plot.svg", format="svg", dpi=300)
    plt.show()

    # Fourth external potentials
    V_ext_four = np.zeros((int(total_time_steps), num_compartments))

    ## Pulse starting at 5ms with an amplitude of -0.15 mA and 1ms long, bi-phasic
    start_time_pulse = int((5e-3 / dt) - 1)
    mid_time_pulse = int(6e-3 / dt)
    end_time_pulse = int(7e-3 / dt)
    fourth_stim_current = -0.15e-3

    ## Calculate and store external potentials
    for index in range(start_time_pulse, end_time_pulse):
        if index == mid_time_pulse - 1:
            fourth_stim_current = -fourth_stim_current
        V_ext_four[index], _ = calculate_potentials_over_distance(rho_medium,
                                                                   fourth_stim_current,
                                                                   axon_length,
                                                                   compartment_length,
                                                                   distance_to_axon)

    ## Simulate neuron using external potentials
    fourth_hh_volt_array, fourth_hh_gates_array, fourth_hh_ion_currents_array, fourth_hh_time_array = hh_model(
        V_ext_four,
        t_end,
        dt,
        temp,
        g_sodium,
        volt_sodium,
        g_potassium,
        volt_potassium,
        g_leak,
        volt_leak,
        capacitance_membrane,
        rho_axon,
        compartment_length,
        radius_axon,
        V_rest=volt_rest,
        compartment_size=num_compartments,
        c_matrix_mode="full"
    )

    np.save("data/fourth_hh_volt_array.npy", fourth_hh_volt_array)
    np.save("data/fourth_hh_time_array.npy", fourth_hh_time_array)
    # Plot using pcolormesh
    time_edges = compute_pcolormesh_edges(fourth_hh_time_array)
    compartments = np.arange(num_compartments)
    compartment_edges = compute_pcolormesh_edges(compartments)
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot with rasterization
    mesh = ax.pcolormesh(
        time_edges,
        compartment_edges,
        fourth_hh_volt_array.T * 1000,
        shading='auto',
        cmap='viridis',
        vmin=-20,
        vmax=100,
        rasterized=True
    )
    # Axis formatting
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Compartment Nr.', fontsize=14)
    ax.set_title('Propagation of action potentials with a bi-phasic pulse stimulation $I_4 = -0.15mA$ at t = 5ms')
    # Format tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000:.1f}'))
    ax.tick_params(axis='both', labelsize=14)
    ax.invert_yaxis()
    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Voltage (mV)')
    # Save as compressed SVG
    fig.savefig("plots/fourth_voltages_plot.svg", format="svg", dpi=300)
    plt.show()

    # Fifth external potentials
    V_ext_five = np.zeros((int(total_time_steps), num_compartments))

    ## Pulse starting at 5ms with an amplitude of 0.2 mA and 1ms long, mono-phasic
    start_time_pulse = int((5e-3 / dt) - 1)
    end_time_pulse = int(6e-3 / dt)
    fifth_stim_current = 0.2e-3

    ## Calculate and store external potentials
    for index in range(start_time_pulse, end_time_pulse):
        V_ext_five[index], _ = calculate_potentials_over_distance(rho_medium,
                                                                 fifth_stim_current,
                                                                 axon_length,
                                                                 compartment_length,
                                                                 distance_to_axon)

    ## Simulate neuron using external potentials
    fifth_hh_volt_array, fifth_hh_gates_array, fifth_hh_ion_currents_array, fifth_hh_time_array = hh_model(
        V_ext_five,
        t_end,
        dt,
        temp,
        g_sodium,
        volt_sodium,
        g_potassium,
        volt_potassium,
        g_leak,
        volt_leak,
        capacitance_membrane,
        rho_axon,
        compartment_length,
        radius_axon,
        V_rest=volt_rest,
        compartment_size=num_compartments,
        c_matrix_mode="full"
    )

    np.save("data/fifth_hh_volt_array.npy", fifth_hh_volt_array)
    np.save("data/fifth_hh_time_array.npy", fifth_hh_time_array)
    # Plot using pcolormesh
    time_edges = compute_pcolormesh_edges(fifth_hh_time_array)
    compartments = np.arange(num_compartments)
    compartment_edges = compute_pcolormesh_edges(compartments)
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot with rasterization
    mesh = ax.pcolormesh(
        time_edges,
        compartment_edges,
        fifth_hh_volt_array.T * 1000,
        shading='auto',
        cmap='viridis',
        vmin=-20,
        vmax=100,
        rasterized=True
    )
    # Axis formatting
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Compartment Nr.', fontsize=14)
    ax.set_title('Propagation of action potentials with a mono-phasic pulse stimulation $I_5 = 0.2mA$ at t = 5ms')
    # Format tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000:.1f}'))
    ax.tick_params(axis='both', labelsize=14)
    ax.invert_yaxis()
    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Voltage (mV)')
    # Save as compressed SVG
    fig.savefig("plots/fifth_voltages_plot.svg", format="svg", dpi=300)
    plt.show()

    # Sixth external potentials
    V_ext_six = np.zeros((int(total_time_steps), num_compartments))

    ## Pulse starting at 5ms with an amplitude of 0.4 mA and 1ms long, mono-phasic
    start_time_pulse = int((5e-3 / dt) - 1)
    end_time_pulse = int(6e-3 / dt)
    sixth_stim_current = 0.4e-3

    ## Calculate and store external potentials
    for index in range(start_time_pulse, end_time_pulse):
        V_ext_six[index], _ = calculate_potentials_over_distance(rho_medium,
                                                                 sixth_stim_current,
                                                                 axon_length,
                                                                 compartment_length,
                                                                 distance_to_axon)

    ## Simulate neuron using external potentials
    sixth_hh_volt_array, sixth_hh_gates_array, sixth_hh_ion_currents_array, sixth_hh_time_array = hh_model(
        V_ext_six,
        t_end,
        dt,
        temp,
        g_sodium,
        volt_sodium,
        g_potassium,
        volt_potassium,
        g_leak,
        volt_leak,
        capacitance_membrane,
        rho_axon,
        compartment_length,
        radius_axon,
        V_rest=volt_rest,
        compartment_size=num_compartments,
        c_matrix_mode="full"
    )

    np.save("data/sixth_hh_volt_array.npy", sixth_hh_volt_array)
    np.save("data/sixth_hh_time_array.npy", sixth_hh_time_array)
    # Plot using pcolormesh
    time_edges = compute_pcolormesh_edges(sixth_hh_time_array)
    compartments = np.arange(num_compartments)
    compartment_edges = compute_pcolormesh_edges(compartments)
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot with rasterization
    mesh = ax.pcolormesh(
        time_edges,
        compartment_edges,
        sixth_hh_volt_array.T * 1000,
        shading='auto',
        cmap='viridis',
        vmin=-20,
        vmax=100,
        rasterized=True
    )
    # Axis formatting
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Compartment Nr.', fontsize=14)
    ax.set_title('Propagation of action potentials with a mono-phasic pulse stimulation $I_6 = 0.4mA$ at t = 5ms')
    # Format tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000:.1f}'))
    ax.tick_params(axis='both', labelsize=14)
    ax.invert_yaxis()
    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Voltage (mV)')
    # Save as compressed SVG
    fig.savefig("plots/sixth_voltages_plot.svg", format="svg", dpi=300)
    plt.show()









