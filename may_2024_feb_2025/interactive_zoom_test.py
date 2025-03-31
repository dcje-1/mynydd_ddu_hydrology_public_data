# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Cursor
# from matplotlib import pyplot as plt
# import matplotlib as mpl

# # Generate synthetic time domain data
# np.random.seed(0)
# time = np.linspace(0, 10, 1000)  # Time vector
# signal = np.sin(2 * np.pi * 0.2 * time) + 0.5 * np.random.randn(len(time))  # Example signal with noise

# # Create the figure and axis objects
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# # Left subplot: Time domain plot
# ax1.plot(time, signal, label="Signal")
# ax1.set_title("Time Domain")
# ax1.set_xlabel("Time")
# ax1.set_ylabel("Amplitude")
# ax1.grid(True)

# # Right subplot: Histogram of the time domain signal
# ax2.hist(signal, bins=30, color='orange', alpha=0.7)
# ax2.set_title("Histogram of Signal")
# ax2.set_xlabel("Amplitude")
# ax2.set_ylabel("Frequency")
# ax2.grid(True)

# # Define a function to update the histogram based on the zoom range of the time domain plot
# def update_histogram(val):
#     # Get the current xlim from the time domain plot
#     xmin, xmax = ax1.get_xlim()

#     # Filter the data to match the zoomed-in time range
#     zoomed_data = signal[(time >= xmin) & (time <= xmax)]
    
#     # Clear the histogram and plot again
#     ax2.clear()
#     ax2.hist(zoomed_data, bins=30, color='orange', alpha=0.7)
#     ax2.set_title("Histogram of Signal")
#     ax2.set_xlabel("Amplitude")
#     ax2.set_ylabel("Frequency")
#     ax2.grid(True)
#     fig.canvas.draw_idle()  # Redraw the figure to update the plot

# # Connect the zoom interaction from ax1 to update the histogram
# ax1.callbacks.connect('xlim_changed', update_histogram)

# # Interactive zoom tool
# mpl.rcParams['toolbar'] = 'None'  # Disable the toolbar for better zoom experience

# # Show the plot
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import matplotlib.dates as mdates

# Generate synthetic time domain data with datetime index
np.random.seed(0)
time = pd.date_range(start='2025-01-01', periods=1000, freq='T')  # Datetime index
signal = np.sin(2 * np.pi * 0.2 * np.arange(len(time))) + 0.5 * np.random.randn(len(time))  # Signal with noise

# Create a DataFrame with datetime index and signal
df = pd.DataFrame({'signal': signal}, index=time)

# Create the figure and axis objects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# Left subplot: Time domain plot
ax1.plot(df.index, df['signal'], label="Signal")
ax1.set_title("Time Domain")
ax1.set_xlabel("Time")
ax1.set_ylabel("Amplitude")
ax1.grid(True)

# Right subplot: Histogram of the time domain signal
hist_data, bins, patches = ax2.hist(df['signal'], bins=30, color='orange', alpha=0.7)
ax2.set_title("Histogram of Signal")
ax2.set_xlabel("Amplitude")
ax2.set_ylabel("Frequency")
ax2.grid(True)

# Define a function to update the histogram based on the zoom range of the time domain plot
def update_histogram(val):
    # Get the current xlim from the time domain plot (in numerical format)
    xmin, xmax = ax1.get_xlim()

    # Convert the numerical xlim values to datetime64 using pandas
    xmin = pd.to_datetime(xmin, unit='D', origin=pd.Timestamp('1970-01-01'))  # Convert to datetime64
    xmax = pd.to_datetime(xmax, unit='D', origin=pd.Timestamp('1970-01-01'))  # Convert to datetime64

    # Filter the data to match the zoomed-in time range
    zoomed_data = df[(df.index >= xmin) & (df.index <= xmax)]['signal']

    # If no data is available in the zoomed range, skip the update
    if len(zoomed_data) == 0:
        return

    # Update the histogram with the new data
    ax2.clear()
    ax2.hist(zoomed_data, bins=30, color='orange', alpha=0.7)
    ax2.set_title("Histogram of Signal")
    ax2.set_xlabel("Amplitude")
    ax2.set_ylabel("Frequency")
    ax2.grid(True)
    
    # Redraw the figure to update the plot
    fig.canvas.draw_idle()

# Connect the zoom interaction from ax1 to update the histogram
ax1.callbacks.connect('xlim_changed', update_histogram)

# Interactive zoom tool
mpl.rcParams['toolbar'] = 'None'  # Disable the toolbar for better zoom experience

# Show the plot
plt.tight_layout()
plt.show()



# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import numpy as np
# import datetime

# # Create some time-based data
# times = [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 10), datetime.datetime(2020, 1, 20)]
# values = [1, 2, 3]

# # Create a plot
# fig, ax = plt.subplots()
# ax.plot(times, values)

# # Set x-axis to handle dates
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# # Get the current x-axis limits (in numerical format)
# xlim = ax.get_xlim()

# # Convert the numerical xlim values to datetime
# start_date = mdates.num2date(xlim[0])
# end_date = mdates.num2date(xlim[1])

# # Print the results
# print("Start Date:", start_date)
# print("End Date:", end_date)

# plt.show()
