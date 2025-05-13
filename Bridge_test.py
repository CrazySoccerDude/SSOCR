from Phidget22.Phidget import *
from Phidget22.Devices.VoltageRatioInput import *
import time
import matplotlib.pyplot as plt
from collections import deque

def read_voltage_ratio_from_channel_0():
    """Reads the voltage ratio from a VoltageRatioInput device on channel 0 and plots it dynamically with a 100s rolling window."""

    # Create a VoltageRatioInput object
    voltage_ratio_input = VoltageRatioInput()

    # Set the channel to 0
    voltage_ratio_input.setChannel(2)  # Specify the channel

    # Open the device and wait for attachment with a timeout
    try:
        voltage_ratio_input.openWaitForAttachment(1000)  # 1000ms timeout
    except PhidgetException as e:
        print(f"Error: {e}")
        return

    # Initialize the plot
    plt.ion()
    fig, ax = plt.subplots()
    time_data = deque(maxlen=1000)  # Store up to 1000 points (assuming 10 points per second for 100 seconds)
    voltage_data = deque(maxlen=1000)
    line, = ax.plot([], [], label="Voltage Ratio")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage Ratio")
    ax.legend()
    start_time = time.time()

    try:
        print("Press Ctrl+C to stop.")
        while True:
            # Get the current voltage ratio
            voltage_ratio = voltage_ratio_input.getVoltageRatio()
            current_time = time.time() - start_time

            # Update data
            time_data.append(current_time)
            voltage_data.append(voltage_ratio)

            # Update the plot
            line.set_data(time_data, voltage_data)
            ax.set_xlim(max(0, current_time - 10), current_time)  # Set x-axis to show the last 100 seconds
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.1)  # Pause to update the plot

    except KeyboardInterrupt:
        print("Reading stopped by user.")
    except PhidgetException as e:
        print(f"Error: {e}")

    finally:
        # Close the device and finalize the plot
        voltage_ratio_input.close()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    read_voltage_ratio_from_channel_0()