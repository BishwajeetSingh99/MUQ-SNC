'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_ignition_delay(csv_file, output_file="ignition_delay.png"):
    # Read CSV
    df = pd.read_csv(csv_file)

    # Extract columns
    T = df["T"].values
    obs = df["Obs(us)"].values
    nom = df["Nominal"].values

    # X-axis as 1000/T
    x = 1000.0 / T

    # Sort nominal values for smooth curve
    order = np.argsort(x)
    x_nom, y_nom = x[order], nom[order]

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_yscale("log")

    # Nominal as curve
    ax.plot(x_nom, y_nom, "-", color="orange", linewidth=1.5, label="Nominal")

    # Obs as scatter
    ax.plot(x, obs, "x", color="black", markersize=6, label="Obs")

    # Labels
    ax.set_xlabel("1000/T, [1/K]", fontsize=12)
    ax.set_ylabel(r"$\tau_{ig}\; [\mu s]$", fontsize=12)

    # Axis limits (matching your reference style)
    ax.set_xlim(0.6, 1.6)
    ax.set_ylim(1e0, 1e5)  # since values are in µs, adjust range if needed

    # Grid
    ax.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.minorticks_on()
    ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.5)

    # Legend
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()


# Example:
# plot_ignition_delay("01322afc-bdbb-47ce-9447-e747a8f8951f.csv", "plot.png")

plot_ignition_delay("/home/user/Desktop/butane_data/kerosean/kerosean_trial/plot/Dataset/Tig/all_tig_data.csv", "ig_plot.png")
'''




import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def plot_converted_data(file_path, output_filename='temperature_plot_ms_reciprocal.png'):
    """
    Reads a CSV file, performs time conversions (us to ms),
    calculates 1000/T for the new x-axis, and plots the data.
    Uses cubic interpolation to draw a smooth curve for Nominal(ms).

    Args:
        file_path (str): The path to the input CSV file.
        output_filename (str): The name for the output plot file.
    """
    try:
        # Load the data from the CSV file
        df = pd.read_csv(file_path)
        
        # IMPORTANT: Sort data by T (and thus Reciprocal_T) for correct interpolation
        df = df.sort_values(by='T').reset_index(drop=True)

        # 1. ADD NEW X-AXIS: 1000/T
        df['Reciprocal_T'] = 1000.0 / df['T']

        # 2. CONVERSION: Convert data to milliseconds
        df['Obs(ms)'] = df['Obs(us)'] / 1e3
        df['Nominal(ms)'] = df['Nominal'] / 1e3

        # 3. SMOOTHING THE NOMINAL LINE (The blue line)
        x_data = df['Reciprocal_T']
        y_nominal_data = df['Nominal(ms)']

        # Create a cubic interpolation function to fit a smooth curve
        f_nominal = interp1d(x_data, y_nominal_data, kind='cubic')

        # Generate a dense array of new x-values for the smooth curve
        x_new = np.linspace(x_data.min(), x_data.max(), 500)
        
        # Generate the smoothed y-values
        y_nominal_smooth = f_nominal(x_new)

        # 4. PLOTTING
        plt.figure(figsize=(10, 6))

        # Plot 'Obs(ms)' vs 1000/T (discrete markers)
        plt.plot(
            df['Reciprocal_T'],
            df['Obs(ms)'],
            marker='v',
            linestyle='',
            label='Obs(ms)',
            color='red',
            markersize=6
        )

        # Plot the SMOOTHED 'Nominal(ms)' curve
        plt.plot(
            x_new, # Use the dense x-values
            y_nominal_smooth, # Use the smoothed y-values
            linestyle='-',
            label='Nominal(ms) (Smoothed)',
            color='blue',
            linewidth=2
        )

        # Add labels and title
        plt.xlabel('$1000/T$') 
        plt.ylabel('Value (ms)')
        plt.title('Obs(ms) and Nominal(ms) vs. $1000/T$')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plt.savefig(output_filename)
        plt.close()

        print(f"Plot successfully saved as {output_filename}")
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage (assuming your CSV is named 'data.csv')
# plot_converted_data('data.csv')


# --- Example Usage ---
# NOTE: Replace 'your_data_file.csv' with the actual path to your data file.
# If you are using the sample data I created earlier, the file is named 'data.csv'.
plot_converted_data('/home/user/Desktop/butane_data/propane_mechanims_fuel_paper/NOMINAL_CASE_PROPANE/Plot/Dataset/Tig/x10001001.csv', '1001')
plot_converted_data('/home/user/Desktop/butane_data/propane_mechanims_fuel_paper/NOMINAL_CASE_PROPANE/Plot/Dataset/Tig/x10001002.csv', '1002')
plot_converted_data('/home/user/Desktop/butane_data/propane_mechanims_fuel_paper/NOMINAL_CASE_PROPANE/Plot/Dataset/Tig/x10001003.csv', '1003')
plot_converted_data('/home/user/Desktop/butane_data/propane_mechanims_fuel_paper/NOMINAL_CASE_PROPANE/Plot/Dataset/Tig/x10001004.csv', '1004')
