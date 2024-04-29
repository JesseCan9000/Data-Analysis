import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import math

def plot_current_flow_rate_and_temperature(squidstat_data_csv, mfc_data_csv, temperature_data_csv):
    '''Creates three stacked plots with a common time axis: Current (nA/cm2), Mass Flow Rates (sccm), and Temperature (degC) vs Time (min) '''

    # Read data from CSV files
    mfc_data = pd.read_csv(mfc_data_csv)
    squidstat_data = pd.read_csv(squidstat_data_csv)
    temperature_data = pd.read_csv(temperature_data_csv, names=['Time', 'Temperature'])
    temperature_data['Time'] = pd.to_datetime(temperature_data['Time'])

    # Extract initial time from the file name
    file_name = squidstat_data_csv.split('/')[-1]
    initial_time_str = file_name.split("(")[-1].split(")")[0]

    # Convert initial time to datetime object
    initial_time = pd.to_datetime(initial_time_str, format="%Y-%m-%d %H_%M_%S")

    # Convert Time column in mfc_data to datetime object
    mfc_data['Time'] = pd.to_datetime(mfc_data['Time'])

    # Calculate elapsed time relative to the initial time for squidstat_data
    squidstat_data['Time'] = initial_time + pd.to_timedelta(squidstat_data['Elapsed Time (s)'], unit='s')
    squidstat_data['Current (nA/cm2)'] = squidstat_data['Current (A)']*math.pow(10, 9)/DEVICE_AREA

    # Merge squidstat_data and mfc_data based on the nearest time
    merged_data = pd.merge_asof(squidstat_data, mfc_data, on='Time', direction='nearest')

    # Convert Elapsed Time to minutes
    merged_data['Elapsed Time (min)'] = merged_data['Elapsed Time (s)'] / 60

    # Merge temperature data with merged_data based on the nearest time
    merged_data = pd.merge_asof(merged_data, temperature_data, on='Time', direction='nearest')

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Current (A) vs. Time
    axs[0].plot(merged_data['Elapsed Time (min)'], merged_data['Current (nA/cm2)'], color='tab:blue')
    axs[0].set_ylabel('Current (nA/cm2)')

    # Plot 2: Flow Rates vs. Time
    axs[1].plot(merged_data['Elapsed Time (min)'], merged_data['Low_Mass_Flow_SCCM'], color='tab:red', label='Ar') # , linestyle='--')
    axs[1].plot(merged_data['Elapsed Time (min)'], merged_data['Middle_Mass_Flow_SCCM'], color='tab:orange', label='5% NH3 in N2') # , linestyle='--')
    axs[1].plot(merged_data['Elapsed Time (min)'], merged_data['High_Mass_Flow_SCCM'], color='tab:green', label='NH3') # , linestyle='--')
    axs[1].set_ylabel('Flow Rates (sccm)')
    axs[1].legend(loc='lower right')

    # Plot 3: Temperature vs. Time
    axs[2].plot(merged_data['Elapsed Time (min)'], merged_data['Temperature'], color='tab:purple') # , linestyle='-.')
    axs[2].set_xlabel('Elapsed Time (min)')
    axs[2].set_ylabel('Temperature (°C)')

    plt.tight_layout()

    def on_xlims_change(event):
        for ax in axs:
            ax.set_xlim(event.get_xlim())
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('xlim_changed', on_xlims_change)

    plt.show()

if __name__ == "__main__":
    # USER: MODIFY THESE LINES TO BE THE FILE PATH OF THE CORRESPODING DATA FILES
    mfc_data_csv = "MFC Log NH3 IPT 02152024.csv" # required for plot_current_flow_rate_and_temperature()
    squidstat_data_csv = "Manual Experiment 0V overnight (2024-02-15 22_08_50)/Manual Experiment(2024-02-15 22_08_50).csv" # required for calculate_peak_areas() and plot_current_flow_rate_and_temperature()
    temperature_data_csv = "Temperature Log NH3 IPT 02152024.csv" # required for plot_current_flow_rate_and_temperature()

    DEVICE_AREA = 1.1 # cm2

    plot_current_flow_rate_and_temperature(squidstat_data_csv, mfc_data_csv, temperature_data_csv)