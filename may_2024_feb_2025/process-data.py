#!/usr/bin/python

# Script to read rainfall and river level CSV files and plot the results
# Imports data with a datetime index
# Date: 08/02/2025

from __future__ import print_function

# import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.widgets import MultiCursor
import scipy.signal as sp
from scipy.signal import savgol_filter, find_peaks, resample, argrelextrema
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib as mpl


# import xlsxwriter


def import_nrw_data(file, start_row, end_row):
    # Import NRW data with the date time parsed as the index
    nrw_df = pd.read_csv(file, skiprows=start_row, nrows=end_row, encoding='utf-8', header=0,
                         names=['Date (UTC)', 'Values'], index_col=0, parse_dates=True)

    date_range = pd.date_range(start=nrw_df.index.min(), end=nrw_df.index.max(), freq='15min')
    nrw_df_regular = nrw_df.reindex(date_range)

    return nrw_df_regular


def import_logger_data(file, start_row, end_row):
    logger_df = pd.read_csv(file, skiprows=start_row, nrows=end_row, encoding='iso-8859-1',
                            header=0, names=['Date/time', 'Water head[cm]', 'Temperature[°C]'], index_col=0,
                            parse_dates=True)

    logger_df['Water head[cm]'] = logger_df['Water head[cm]'] / 100
    logger_df.rename(columns={'Water head[cm]': 'Water head[m]'}, inplace=True)

    return logger_df


def identify_peaks(logger_data):
    # level_peak_times_idx, _ = sp.find_peaks(logger_data['Water head[m]'], distance=10, prominence=0.03)
    level_peak_times_idx, _ = sp.find_peaks(logger_data['Water head[m]'], distance=50, prominence=0.05)
    temperature_peak_times_idx, _ = sp.find_peaks(logger_data['Temperature[°C]'], distance=50, prominence=0.04)
    temperature_trough_times_idx, _ = sp.find_peaks(-logger_data['Temperature[°C]'], distance=10, prominence=0.05)

    # level_peak_times_idx = []
    # temperature_peak_times_idx = []
    # temperature_trough_times_idx = []

    return level_peak_times_idx, temperature_peak_times_idx, temperature_trough_times_idx


def calculate_averages(level_values):
    ## Calculate the N hour rolling mean before each level value
    #  1 sample = 5 minutes. Rolling() defaults to take the previous N samples,
    #  therefore rolling(4).mean() takes the current sample and the previous 3
    #  and then calculates the mean, and places that in the current position
    one_hr_mean = level_values['Water head[m]'].rolling(12).mean()
    two_hr_mean = level_values['Water head[m]'].rolling(24).mean()
    four_hr_mean = level_values['Water head[m]'].rolling(48).mean()
    six_hr_mean = level_values['Water head[m]'].rolling(72).mean()
    eight_hr_mean = level_values['Water head[m]'].rolling(96).mean()
    twelve_hr_mean = level_values['Water head[m]'].rolling(144).mean()
    twentyfour_hr_mean = level_values['Water head[m]'].rolling(288).mean()
    fourtyeight_hr_mean = level_values['Water head[m]'].rolling(576).mean()
    seventytwo_hr_mean = level_values['Water head[m]'].rolling(864).mean()
    ninetysix_hr_mean = level_values['Water head[m]'].rolling(1152).mean()
    onetwenty_hr_mean = level_values['Water head[m]'].rolling(1440).mean()

    return one_hr_mean, two_hr_mean, four_hr_mean, six_hr_mean, \
        eight_hr_mean, twelve_hr_mean, twentyfour_hr_mean, \
        fourtyeight_hr_mean, seventytwo_hr_mean, ninetysix_hr_mean, \
        onetwenty_hr_mean


def plot_details(dyo_level, dyo_level_peak_times_idx, dyo_temperature_peak_times_idx, wff_level,
                 wff_level_peak_times_idx, dyo_onetwenty_hr_mean, nrw_dyo_rainfall, dyo_baro, labels):
    with PdfPages(r'outputs/summary_plots_2.pdf') as export_pdf:
        # Change the default font family to Arial
        plt.rcParams.update({'font.family': 'arial'})
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex='all')
        fig.suptitle('Plot of sink, resurgence & rainfall data,\n highlighting peak depth '
                     'times (and sample numbers)', fontsize=12, weight="bold")

        # Plot rainfall on same plot, dual y axis
        lines0 = ax0.plot(nrw_dyo_rainfall['Values'], '0.8', label=labels[1])
        ax0.invert_yaxis()
        ax0.set_ylabel('Rainfall (mm)', weight="bold")

        ax3 = ax0.twinx()

        # Plot WFF level
        lines1 = ax3.plot(wff_level['Water head[m]'], '-b', label=labels[0])
        # Plot peaks
        ax3.plot(wff_level['Water head[m]'][wff_level_peak_times_idx], 'mo')
        ax3.set_ylabel('Water depth (m)', weight="bold")
        ax3.grid(True)
        # Add peak annotations - sample numbers
        for i in range(0, len(wff_level_peak_times_idx)):
            ax3.annotate(f'wff_P:{wff_level_peak_times_idx[i]}',
                         xy=(wff_level.index[wff_level_peak_times_idx[i]],
                             wff_level['Water head[m]'][wff_level_peak_times_idx[i]]),
                         fontsize=8, color='m', xytext=(
                wff_level.index[wff_level_peak_times_idx[i]], wff_level['Water head[m]'][wff_level_peak_times_idx[i]]))

        lines2 = lines0 + lines1
        labels1 = [l.get_label() for l in lines2]
        ax0.legend(lines2, labels1, loc=0)

        # Plot DYO
        lines3 = ax1.plot(dyo_level['Water head[m]'], '-b', label=labels[2])
        ax1.plot(dyo_level['Water head[m]'][dyo_level_peak_times_idx], 'bo')
        lines4 = ax1.plot(dyo_onetwenty_hr_mean, '-c', label=labels[3])
        ax1.set_ylabel('Water depth (m)', weight="bold")
        ax1.grid(True)

        for i in range(0, len(dyo_level_peak_times_idx)):
            ax1.annotate(f'dyo_P:{dyo_level_peak_times_idx[i]}',
                         xy=(dyo_level.index[dyo_level_peak_times_idx[i]],
                             dyo_level['Water head[m]'][dyo_level_peak_times_idx[i]]),
                         fontsize=8, color='m', xytext=(
                dyo_level.index[dyo_level_peak_times_idx[i]], dyo_level['Water head[m]'][dyo_level_peak_times_idx[i]]))

        ax4 = ax1.twinx()
        lines5 = ax4.plot(dyo_level['Temperature[°C]'], '-r', label=labels[4])
        ax4.plot(dyo_level['Temperature[°C]'][dyo_temperature_peak_times_idx], 'ro')

        for i in range(0, len(dyo_temperature_peak_times_idx)):
            ax4.annotate(f'dyo_T:{dyo_temperature_peak_times_idx[i]}',
                         xy=(dyo_level.index[dyo_temperature_peak_times_idx[i]],
                             dyo_level['Temperature[°C]'][dyo_temperature_peak_times_idx[i]]),
                         fontsize=8, color='m', xytext=(dyo_level.index[dyo_temperature_peak_times_idx[i]],
                                                        dyo_level['Temperature[°C]'][
                                                            dyo_temperature_peak_times_idx[i]]))

        lines6 = ax4.plot(dyo_baro['Temperature[°C]'], '-g', label=labels[5])

        ax4.set_ylabel('Temperature (°C)', weight="bold")

        lines7 = lines3 + lines4 + lines5 + lines6
        labels2 = [l.get_label() for l in lines7]
        ax4.legend(lines7, labels2, loc=0)

        ax1.set_xlabel('Timestamp', weight="bold")
        ax1.tick_params(axis='x', rotation=50)
        xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
        ax1.xaxis.set_major_formatter(xfmt)
        # ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.25))

        # Defining the cursor
        # multi = MultiCursor(fig.canvas, (ax0, ax1, ax3), color='r', lw=1,
        #                     horizOn=False, vertOn=True)

        fig.tight_layout()
        mng = plt.get_current_fig_manager()
        # Resize in pixels
        mng.resize(1500, 720)
        export_pdf.savefig()
        plt.show()


def plot_temperature_detail(dyo_level, dyo_baro):
    plt.rcParams.update({'font.family': 'arial'})
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [3, 1, 1]})
    fig.suptitle('Plot of Dan yr Ogof river cave water and air temperature details', fontsize=12, weight="bold")

    line0 = ax0.plot(dyo_level['Temperature[°C]'], '-r', label='DYO water temperature', linewidth=0.8)
    line1 = ax0.plot(dyo_baro['Temperature[°C]'], '-b', label='DYO air temperature', linewidth=0.8)
    ax0.set_ylabel('Temperature, °C')
    ax0.set_xlabel('Date/Time')
    ax0.tick_params(axis='x', rotation=50)
    xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
    ax0.xaxis.set_major_formatter(xfmt)
    # ax0.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax0.grid()

    ax3 = ax0.twinx()
    line2 = ax3.plot(dyo_level['Water head[m]'], '-m', label='DYO water depth', linewidth=0.8)
    ax3.set_ylabel('Water depth, m')

    lines = line0 + line1 + line2
    labels = [l.get_label() for l in lines]
    ax0.legend(lines, labels, loc=0)

    # Calculate the standard deviation
    level_std_dev = np.std(dyo_level['Temperature[°C]'])
    baro_std_dev = np.std(dyo_baro['Temperature[°C]'])

    ax1.hist(dyo_level['Temperature[°C]'], bins=100, color='r', orientation='horizontal', label='DYO water temperature')
    ax1.set_xlabel('Distribution (100 bins)')
    ax1.grid()

    ax2.hist(dyo_baro['Temperature[°C]'], bins=100, color='b', orientation='horizontal', label='DYO air temperature')
    ax2.set_xlabel('Distribution (100 bins)')
    ax2.grid()

    # Annotate the standard deviation on the plot
    ax1.text(0.95, 0.95, f'Standard Deviation: {level_std_dev:.2f}',
             ha='right', va='top', transform=ax1.transAxes,
             fontsize=8, fontname='Arial', color='black')

    ax2.text(0.95, 0.95, f'Standard Deviation: {baro_std_dev:.2f}',
             ha='right', va='top', transform=ax2.transAxes,
             fontsize=8, fontname='Arial', color='black')


    # Function to update histogram based on the zoom region of the time plot
    def update_histogram(event):
        # Get the visible x-limits of the left plot (ax1)
        xlim = ax0.get_xlim()

        # Convert xlim values to datetime if needed (since ax1 is datetime-indexed)
        start_date = pd.to_datetime(xlim[0], unit='D', origin=pd.Timestamp('1970-01-01'))
        end_date = pd.to_datetime(xlim[1], unit='D', origin=pd.Timestamp('1970-01-01'))

        # Filter the signal data based on the current x-axis limits
        zoomed_level_data = dyo_level[(dyo_level.index >= start_date) & (dyo_level.index <= end_date)][
            "Temperature[°C]"]
        zoomed_baro_data = dyo_baro[(dyo_baro.index >= start_date) & (dyo_baro.index <= end_date)]["Temperature[°C]"]

        # Disable the callbacks temporarily to avoid recursive calls
        ax0.callbacks.disconnect('xlim_changed')  # Disable the callback temporarily

        # Clear the axes before redrawing
        ax1.clear()  # Clear the current histogram from ax1
        ax2.clear()  # Clear the current histogram from ax2

        ax1.hist(zoomed_level_data, bins=100, color='r', orientation='horizontal',
                 label='DYO water temperature')
        ax1.set_xlabel('Distribution (100 bins)')
        ax1.grid()

        ax2.hist(zoomed_baro_data, bins=100, color='b', orientation='horizontal',
                 label='DYO air temperature')
        ax2.set_xlabel('Distribution (100 bins)')
        ax2.grid()

        # Calculate the standard deviation
        level_std_dev_new = np.std(zoomed_level_data)
        baro_std_dev_new = np.std(zoomed_baro_data)

        # Annotate the standard deviation on the plot
        ax1.text(0.95, 0.95, f'Standard Deviation: {level_std_dev_new:.2f}',
                 ha='right', va='top', transform=ax1.transAxes,
                 fontsize=8, fontname='Arial', color='black')

        ax2.text(0.95, 0.95, f'Standard Deviation: {baro_std_dev_new:.2f}',
                 ha='right', va='top', transform=ax2.transAxes,
                 fontsize=8, fontname='Arial', color='black')

        # Redraw the plot
        fig.canvas.draw_idle()

    # Connect the zoom/pan event to the update function for the time plot's xlim
    # Disable the callback temporarily before the figure is shown
    ax0.callbacks.disconnect('xlim_changed')

    # Use a flag to avoid the initial callback call
    def on_figure_shown(event):
        # Reconnect the callback only after the figure has been fully initialized
        ax0.callbacks.connect('xlim_changed', update_histogram)
        fig.canvas.mpl_disconnect(figure_shown_connection)  # Disconnect after it's been handled

    # Connect to the 'draw_event' to trigger when the figure is drawn for the first time
    figure_shown_connection = fig.canvas.mpl_connect('draw_event', on_figure_shown)

    fig.tight_layout()
    fig.savefig('outputs/temperature.pdf')
    plt.show()

def plot_smoothed_temperatures(dyo_level, dyo_baro):
    plt.rcParams.update({'font.family': 'arial'})
    fig, ax0 = plt.subplots()
    fig.suptitle('Plot of Dan yr Ogof river cave water and air temperature details with smoothing filter', fontsize=12, weight="bold")

    water_oth_mean = dyo_level['Temperature[°C]'].rolling(1440).mean()
    air_oth_mean = dyo_baro['Temperature[°C]'].rolling(1440).mean()

    line0 = ax0.plot(water_oth_mean, '-r', label='120 hour mean water temperature', linewidth=1)
    line1 = ax0.plot(air_oth_mean, '-b', label='120 hour mean air temperature', linewidth=1)
    ax0.set_ylabel('Temperature, °C')
    ax0.set_xlabel('Date/Time')
    ax0.tick_params(axis='x', rotation=50)
    xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
    ax0.xaxis.set_major_formatter(xfmt)
    # ax0.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax0.grid()
    ax0.legend()

    # lines = line0 + line1
    # labels = [l.get_label() for l in lines]
    # ax0.legend(lines, labels, loc=0)

    fig.tight_layout()
    fig.savefig('outputs/smoothed_temperature.pdf')
    plt.show()


def calculate_stats(name, data_frame):
    # Calc stats
    mean = np.mean(data_frame['Temperature[°C]'])
    median = np.median(data_frame['Temperature[°C]'])
    std_dev = np.std(data_frame['Temperature[°C]'], ddof=1)
    std_err = np.std(data_frame['Temperature[°C]'], ddof=1) / np.sqrt(np.size(data_frame['Temperature[°C]']))
    minimum = min(data_frame['Temperature[°C]'])
    maximum = max(data_frame['Temperature[°C]'])
    range = maximum - minimum
    CV = np.std(data_frame['Temperature[°C]'], ddof=1) / mean * 100
    num_samps = np.size(data_frame['Temperature[°C]'])

    with open('outputs/' + name + '_stats.txt', "a") as f:
        print(f'{name} temperature mean = {mean:.4f}°C', file=f)
        print(f'{name} temperature median = {median:.4f}°C', file=f)
        print(f'{name} temperature standard deviation = {std_dev:.4f}', file=f)
        print(f'{name} temperature standard error = {std_err:.4f}', file=f)
        print(f'{name} temperature minimum = {minimum:.4f}°C', file=f)
        print(f'{name} temperature maximum = {maximum:.4f}°C', file=f)
        print(f'{name} temperature range = {range:.4f}°C', file=f)
        print(f'{name} temperature coefficient of variation = {CV:.4f}', file=f)
        print(f'{name} temperature number of samples = {num_samps}', file=f)


def main():
    # Name input files
    file1 = "CSV/wff_depth_consolidated.csv"
    file2 = "CSV/dyo_depth_consolidated.csv"
    file3 = "CSV/dyo_baro_consolidated.csv"
    file4 = "CSV/nrw_rain_consolidated.csv"
    # file4 = "CSV/NRW_TAWE_20250208215134.csv"

    ## Import data
    wff_level = import_logger_data(file1, 1, 61915)
    dyo_level = import_logger_data(file2, 1, 80017)
    dyo_baro = import_logger_data(file3, 1, 80017)
    nrw_dyo_rainfall = import_nrw_data(file4, 1, 26786)

    ## Identify peaks times
    dyo_level_peak_times_idx, dyo_temperature_peak_times_idx, dyo_temperature_trough_times_idx = identify_peaks(
        dyo_level)
    wff_level_peak_times_idx, wff_temperature_peak_times_idx, wff_temperature_trough_times_idx = identify_peaks(
        wff_level)

    ## Calculate the average level at the resurgence before each flood pulse - background levels
    (dyo_one_hr_mean, dyo_two_hr_mean, dyo_four_hr_mean, dyo_six_hr_mean, dyo_eight_hr_mean, dyo_twelve_hr_mean,
     dyo_twentyfour_hr_mean, dyo_fourtyeight_hr_mean, dyo_seventytwo_hr_mean, dyo_ninetysix_hr_mean,
     dyo_onetwenty_hr_mean) = calculate_averages(dyo_level)

    # Calculate stats and write to file
    # calculate_stats('DYO', dyo_level)
    # calculate_stats('WFF', wff_level)
    # calculate_stats('DYO Baro', dyo_baro)

    # plot_temperature_detail(dyo_level, dyo_baro)

    ## Plot each sink against rainfall & resurgence, and highlight peak/rise samples
    # labels = ['WFF water depth', 'NRW rainfall', 'DYO water depth', 'DYO 120 hour water depth moving mean',
    #           'DYO water temperature', 'DYO air temperature']
    # plot_details(dyo_level, dyo_level_peak_times_idx, dyo_temperature_peak_times_idx, wff_level, wff_level_peak_times_idx, dyo_onetwenty_hr_mean, nrw_dyo_rainfall, dyo_baro, labels)

    plot_smoothed_temperatures(dyo_level, dyo_baro)


if __name__ == "__main__":
    main()
