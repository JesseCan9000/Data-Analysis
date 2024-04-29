import pandas as pd
import matplotlib.pyplot as plt

# The things you need to change
file_path = '5sccm_Ar_to_25sccm_NH3_BT_04242024.txt'
analyte = 'NH3_17 (Torr)'
p_inlet = 760 # torr
p_bypass = 8.690E-8 # torr
Q_sccm = 25 # sccm
R = 62.36367 # L*torr/mol K
T = 298 # K

################################ Functions ################################

def calibrate_concentration(C):
    chi = p_inlet/p_bypass
    return chi*C

def n_ads(Q, p_bypass, df):
    ''' Q = flow rate (sccm), p_bypass = gas of interest mass spec partial 
    pressure (torr), df = pandas dataframe containing pressure vs time data'''

    Q_Ls = Q/(60*1000) # L/s
    C_0 = p_bypass/(R*T) # mol/L
    C_0_prime = calibrate_concentration(C_0)

    integral = 0
    for i in range(len(df['Elapsed Time (s)'])):
        if i == 0:
            integral = integral + (1 - df.loc[i, 'NH3_17 (Torr)']/p_bypass)*(df.loc[i, 'Elapsed Time (s)'] - 0)
        else:
            integral = integral + (1 - df.loc[i, 'NH3_17 (Torr)']/p_bypass)*(df.loc[i, 'Elapsed Time (s)'] - df.loc[i-1, 'Elapsed Time (s)'])

    n_ads = Q_Ls*C_0_prime*integral # mol

    return n_ads

################################ Read Data ################################

# Read the lines from the text file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Find the start of the data
start_index = None
for i, line in enumerate(lines):
    if "Elapsed Time" in line:
        start_index = i
        break

# Read the data into a DataFrame
df = pd.read_csv(file_path, skiprows=start_index, delimiter=',', skipinitialspace=True)

# Smooth out the noisy data for each series using a moving average
window_size = 10  # Adjust the window size as needed
smoothed_df = df.rolling(window=window_size).mean()
smoothed_df.to_csv('smooth.csv')

################################ Data Analysis ################################

print(n_ads(Q_sccm, p_bypass, df))

################################ Plotting ################################

# Plot both raw and smoothed data on separate figures
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

# Plot raw data
for column in df.columns[1:]:  # Exclude the first column (Elapsed Time)
    axes[0].plot(df['Elapsed Time (s)'], df[column], label=column)

axes[0].set_xlabel('Elapsed Time (s)')
axes[0].set_ylabel('Pressure (Torr)')
axes[0].set_title('Raw Derived Pressure vs. Time Data')
axes[0].grid(True)

# Plot smoothed data
for column in smoothed_df.columns[1:]:  # Exclude the first column (Elapsed Time)
    axes[1].plot(smoothed_df['Elapsed Time (s)'], smoothed_df[column], label=column)

axes[1].set_xlabel('Elapsed Time (s)')
axes[1].set_ylabel('Pressure (Torr)')
axes[1].set_title(f'Smoothed Derived Pressure vs. Time Data')
axes[1].grid(True)

# Move the legend below the subplots and make it shared
# plt.subplots_adjust(bottom=1.8)
fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=smoothed_df.shape[1]-1)
plt.tight_layout()
plt.show()
