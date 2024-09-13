from datetime import datetime
import time
import keyboard

# Initialize the pressure variable
pressure = 1.6E-6  # Replace this with your actual pressure data source

# Start the logging loop
print("Press spacebar to log pressure. Press 'Ctrl+C' to exit.")

current_time = datetime.now().strftime("%H:%M:%S")
print(f"Time: {current_time}, Pressure: {pressure}")

while True:
    # Check if the backslash key is pressed
    keyboard.wait('space')

    # Log the current time and pressure
    current_time = datetime.now().strftime("%H:%M:%S")
    pressure = pressure - (0.1E-6)
    print(f"Time: {current_time}, Pressure: {pressure:.1e}")
        

    # time.sleep(0.5)