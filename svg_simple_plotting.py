import matplotlib.pyplot as plt
import numpy as np

from hh_utils import compute_pcolormesh_edges

time_arr = np.load("results_data/first_hh_time_array.npy")
volt_arr = np.load("results_data/first_hh_volt_array.npy")

compartments_size = volt_arr.shape[1]

# Plot using pcolormesh
time_edges = compute_pcolormesh_edges(time_arr)
compartments = np.arange(compartments_size)
compartment_edges = compute_pcolormesh_edges(compartments)

fig, ax = plt.subplots(figsize=(10, 5))

# Plot with rasterization
mesh = ax.pcolormesh(
    time_edges,
    compartment_edges,
    volt_arr.T * 1000,
    shading='auto',
    cmap='viridis',
    rasterized=True  # This is the key for compression!
)

# Axis formatting
ax.set_xlabel('Time (ms)', fontsize=14)
ax.set_ylabel('Compartment Nr.', fontsize=14)
ax.set_title('Voltage over Time per Compartment for the second stimulus current')

# Format tick labels
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 1000:.1f}'))
ax.tick_params(axis='both', labelsize=14)
ax.invert_yaxis()

# Add colorbar
cbar = plt.colorbar(mesh, ax=ax)
cbar.set_label('Voltage (mV)')

# Save as compressed SVG
fig.savefig("plots/first_voltages_plot_light.svg", format="svg", dpi=300)
plt.show()
