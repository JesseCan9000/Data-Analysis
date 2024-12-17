import pandas as pd
from datetime import datetime

# File paths
date = "11152024"
mfc_path = f"C:/Users/jcmar/OneDrive/Desktop/Data/Isopotential Titration/NH3/0V/{date}/MFC Log NH3 IPT {date}.csv"
peak_start_times_path = "Peak Offsets.csv"


# Load dataframes
mfc_df = pd.read_csv(mfc_path)
peak_start_times_df = pd.read_csv(peak_start_times_path)

# Convert 'Start Time' to a datetime object for easier comparison
def convert_start_time(start_time_str):
    return datetime.strptime(start_time_str, "%H_%M_%S")

for column in peak_start_times_df.columns:
    if column != "Folder":
        # print(peak_start_times_df.at[0, column])
        # date_str = peak_start_times_df.at[0, column]
        date_obj = datetime.strptime(column, "%m%d%Y")
        time_str = peak_start_times_df.at[0, column]
        time_obj = convert_start_time(time_str)
        peak_start_times_df.at[0, column] = datetime.combine(date_obj.date(), time_obj.time())

def convert_cell_to_datetime(cell_value, start_time):
    """Convert a cell value to datetime by adding it to the start time"""
    # Convert cell_value to a float to ensure it's numeric (in case it's a string)
    try:
        numeric_value = float(cell_value)
    except ValueError:
        print(f"Invalid value: {cell_value}")
        return None  # Handle invalid values gracefully

    # Add the numeric value as seconds to the start time
    return start_time + pd.Timedelta(numeric_value, unit='s')

# Iterate through columns to convert values starting from the second row
for column in peak_start_times_df.columns:
    if column != "Folder":  # Skip the "Folder" column
        start_time_str = peak_start_times_df.at[0, column]  # Read the start time (expected as string)
        start_time = pd.to_datetime(start_time_str, format='%H_%M_%S')  # Convert start time to datetime object

        # Apply conversion function to each cell in the column
        peak_start_times_df.loc[1:, column] = peak_start_times_df.loc[1:, column].apply(
            lambda x: convert_cell_to_datetime(x, start_time)
        )

# Check the result
print(peak_start_times_df)
quit()

##### TODO: fix identification of flow rate flips in MFC data and zip the data together between the two dataframes

# Initialize flip_timestamps dictionary
flip_timestamps = {}  # Value is a list, with index 0 being the MFC flip and index 1 being the peak start time

adsorption_peak_counter = 1
desorption_peak_counter = 1
Ar_check = 10
NH3_check = 40
total_flow = 50

# Initialize previous state variables
previous_adsorption = None
previous_desorption = None

# Loop through MFC DataFrame
for index, row in mfc_df.iterrows():
    # Check if adsorption flip condition is met (Low_Mass_Flow < Ar_check and High_Mass_Flow > NH3_check)
    if row['Low_Mass_Flow_Setpoint_SCCM'] <= Ar_check and row['High_Mass_Flow_Setpoint_SCCM'] >= NH3_check:
        # Only add if the state has flipped from a previous desorption or adsorption
        if previous_adsorption != "adsorption":
            # Add MFC flip time to the dictionary
            flip_timestamps[f"Adsorption {adsorption_peak_counter}"] = [row['Time']]
            adsorption_peak_counter += 1
            previous_adsorption = "adsorption"
            NH3_check = total_flow - NH3_check  # Swap NH3 and Ar check values
            Ar_check = total_flow - Ar_check

    # Check if desorption flip condition is met (Low_Mass_Flow > Ar_check and High_Mass_Flow < NH3_check)
    elif row['Low_Mass_Flow_Setpoint_SCCM'] >= Ar_check and row['High_Mass_Flow_Setpoint_SCCM'] <= NH3_check:
        # Only add if the state has flipped from a previous desorption or adsorption
        if previous_desorption != "desorption":
            # Add MFC flip time to the dictionary
            flip_timestamps[f"Desorption {desorption_peak_counter}"] = [row['Time']]
            desorption_peak_counter += 1
            previous_desorption = "desorption"
            NH3_check = total_flow - NH3_check  # Swap NH3 and Ar check values
            Ar_check = total_flow - Ar_check

# Loop through peak_start_times_df to add corresponding peak start times
for column in peak_start_times_df.columns:
    if column != "Start Time":
        for idx, row in peak_start_times_df.iterrows():
            # Add each peak start time to the respective flip timestamp in flip_timestamps
            if column.startswith("Adsorption"):
                peak_name = f"{column} {idx+1}"
                if peak_name in flip_timestamps:
                    flip_timestamps[peak_name].append(row[column])
            elif column.startswith("Desorption"):
                peak_name = f"{column} {idx+1}"
                if peak_name in flip_timestamps:
                    flip_timestamps[peak_name].append(row[column])

# Print out the results
for key, value in flip_timestamps.items():
    print(f"{key}: {value}")
