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


# import xlsxwriter


def import_nrw_data(file, start_row, end_row):
    # Import NRW data with the date time parsed as the index
    nrw_df = pd.read_csv(file, skiprows=start_row, nrows=end_row, encoding='utf-8', header=0,
                         names=['Date (UTC)', 'Values'], index_col=0, parse_dates=True)

    return nrw_df


def import_logger_data(file, start_row, end_row):

    logger_df = pd.read_csv(file, skiprows=start_row, nrows=end_row  , encoding='iso-8859-1',
                            header=0, names=['Date/time','Water head[cm]','Temperature[°C]'], index_col=0, parse_dates=True)

    logger_df['Water head[cm]'] = logger_df['Water head[cm]']/100
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
                 wff_level_peak_times_idx, dyo_onetwenty_hr_mean, nrw_dyo_rainfall, labels):
    with PdfPages(r'outputs/summary_plots_2.pdf') as export_pdf:
        # Change the default font family to Arial
        plt.rcParams.update({'font.family': 'arial'})
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex='all')
        fig.suptitle('Plot of sink, resurgence & rainfall data,\n highlighting peak stage '
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
        ax3.set_ylabel('Stream Stage (m)', weight="bold")
        ax3.grid(True)
        # Add peak annotations - sample numbers
        for i in range(0, len(wff_level_peak_times_idx)):
            ax3.annotate(f'wff_P:{wff_level_peak_times_idx[i]}',
                         xy=(wff_level.index[wff_level_peak_times_idx[i]], wff_level['Water head[m]'][wff_level_peak_times_idx[i]]),
                         fontsize=8, color='m', xytext=(wff_level.index[wff_level_peak_times_idx[i]], wff_level['Water head[m]'][wff_level_peak_times_idx[i]]))

        lines2 = lines0 + lines1
        labels1 = [l.get_label() for l in lines2]
        ax0.legend(lines2, labels1, loc=0)

        # Plot DYO
        lines3 = ax1.plot(dyo_level['Water head[m]'], '-b', label=labels[2])
        ax1.plot(dyo_level['Water head[m]'][dyo_level_peak_times_idx], 'bo')
        lines4 = ax1.plot(dyo_onetwenty_hr_mean, '-c', label=labels[3])
        ax1.set_ylabel('River Stage (m)', weight="bold")
        ax1.grid(True)

        for i in range(0, len(dyo_level_peak_times_idx)):
            ax1.annotate(f'dyo_P:{dyo_level_peak_times_idx[i]}',
                         xy=(dyo_level.index[dyo_level_peak_times_idx[i]], dyo_level['Water head[m]'][dyo_level_peak_times_idx[i]]),
                         fontsize=8, color='m', xytext=(dyo_level.index[dyo_level_peak_times_idx[i]], dyo_level['Water head[m]'][dyo_level_peak_times_idx[i]]))

        ax4 = ax1.twinx()
        lines5 = ax4.plot(dyo_level['Temperature[°C]'], '-r', label=labels[4])
        ax4.plot(dyo_level['Temperature[°C]'][dyo_temperature_peak_times_idx], 'ro')

        for i in range(0, len(dyo_temperature_peak_times_idx)):
            ax4.annotate(f'dyo_T:{dyo_temperature_peak_times_idx[i]}',
                         xy=(dyo_level.index[dyo_temperature_peak_times_idx[i]], dyo_level['Temperature[°C]'][dyo_temperature_peak_times_idx[i]]),
                         fontsize=8, color='m', xytext=(dyo_level.index[dyo_temperature_peak_times_idx[i]], dyo_level['Temperature[°C]'][dyo_temperature_peak_times_idx[i]]))


        ax4.set_ylabel('Temperature (°C)', weight="bold")

        lines6 = lines3 + lines4 + lines5
        labels2 = [l.get_label() for l in lines6]
        ax4.legend(lines6, labels2, loc=0)

        ax1.set_xlabel('Timestamp', weight="bold")
        ax1.tick_params(axis='x', rotation=50)
        xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
        ax1.xaxis.set_major_formatter(xfmt)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.25))

        # Defining the cursor
        # multi = MultiCursor(fig.canvas, (ax0, ax1, ax3), color='r', lw=1,
        #                     horizOn=False, vertOn=True)

        fig.tight_layout()
        mng = plt.get_current_fig_manager()
        # Resize in pixels
        mng.resize(1500, 720)
        export_pdf.savefig()
        plt.show()


def plot_all_levels(nrw_tawe_level, nrw_dyo_rainfall, wff_level, dyo_level):
    with PdfPages('outputs/summary_plots_1.pdf') as export_pdf:
        # Change the default font family to Arial
        plt.rcParams.update({'font.family': 'arial'})

        # Plot results - datetime verses depth
        fig, (ax0, ax1, ax3) = plt.subplots(nrows=3, sharex='all')
        fig.suptitle('Summary plot of the Waun Fignen Felen stream stage\n versus '
                     'the Dan yr Ogof Resurgence and \nTawe river stage',
                     fontsize=12, weight="bold")

        lines0 = ax0.plot(nrw_tawe_level['Values'], 'b', label='NRW Tawe stage at Craig y Nos')
        lines1 = ax0.plot(wff_level['Water head[m]'], 'm', label='WFF stage')
        lines2 = ax0.plot(dyo_level['Water head[m]'], 'c', label='DYO resurgence stage')

        ax0.set_ylabel('Stage (m)', weight="bold")
        ax0.tick_params(axis='y', color='b', labelcolor='k')

        lines3 = ax1.plot(nrw_dyo_rainfall['Values'], '0.8', label='NRW rainfall at DYO')
        ax1.invert_yaxis()
        ax1.legend()

        ax2 = ax1.twinx()
        lines4 = ax3.plot(wff_level['Temperature[°C]'], 'm', label='WFF temperature')
        ax3.legend()
        lines5 = ax2.plot(dyo_level['Temperature[°C]'], 'c', label='DYO temperature')
        ax2.legend()

        lines_1 = lines0 + lines1 + lines2
        labels = [l.get_label() for l in lines_1]
        ax0.legend(lines_1, labels, loc=0)

        # lines_2 = lines4 + lines5
        # labels = [l.get_label() for l in lines_2]
        # ax2.legend(lines_2, labels, loc=0)

        ax1.set_ylabel('Rainfall (mm)', weight="bold")
        ax2.set_ylabel('Water Temperature ($^\circ$C)', weight="bold")
        ax3.set_ylabel('Water Temperature ($^\circ$C)', weight="bold")
        ax3.set_xlabel('Time stamp', weight="bold")
        xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
        ax3.xaxis.set_major_formatter(xfmt)
        ax3.tick_params(axis='x', rotation=50)

        # Adjust x-tick resolution for zooming in
        ax3.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax0.autoscale(enable=True, axis='both', tight=None)
        ax1.autoscale(enable=True, axis='both', tight=None)
        ax3.autoscale(enable=True, axis='both', tight=None)
        ax0.grid(True)
        ax1.grid(True)
        ax3.grid(True)

        fig.tight_layout()
        mng = plt.get_current_fig_manager()
        # Resize in pixels
        mng.resize(1500, 720)
        export_pdf.savefig()
        plt.show()
        #plt.close()


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

    with open('outputs/'+name+'_stats.txt', "a") as f:
        print(f'{name} temperature mean = {mean:.4f}°C', file=f)
        print(f'{name} temperature median = {median:.4f}°C', file=f)
        print(f'{name} temperature standard deviation = {std_dev:.4f}', file=f)
        print(f'{name} temperature standard error = {std_err:.4f}', file=f)
        print(f'{name} temperature minimum = {minimum:.4f}°C', file=f)
        print(f'{name} temperature maximum = {maximum:.4f}°C', file=f)
        print(f'{name} temperature range = {range:.4f}°C', file=f)
        print(f'{name} temperature coefficient of variation = {CV:.4f}', file=f)
        print(f'{name} temperature number of samples = {num_samps}', file=f)


def write_excel(nrw_tawe_level,
               nrw_dyo_rainfall,
               wff_level,
               wff_level_peak_times_idx,
               wff_temperature_peak_times_idx,
               wff_temperature_trough_times_idx,
               giedd_level,
               giedd_level_peak_times_idx,
               giedd_temperature_peak_times_idx,
               giedd_temperature_trough_times_idx,
               dyo_level,
               dyo_level_peak_times_idx,
               dyo_temperature_peak_times_idx,
               dyo_temperature_trough_times_idx,
               dyo_one_hr_mean, dyo_two_hr_mean,
               dyo_four_hr_mean, dyo_six_hr_mean,
               dyo_eight_hr_mean, dyo_twelve_hr_mean, dyo_twentyfour_hr_mean,
               dyo_fourtyeight_hr_mean, dyo_seventytwo_hr_mean,
              dyo_ninetysix_hr_mean, dyo_onetwenty_hr_mean):

    # Add the mean levels to "dyo_level" dataframe
    dyo_level.insert(2, 'One hour moving mean (m)', dyo_one_hr_mean)
    dyo_level.insert(3, 'Two hour moving mean (m)', dyo_two_hr_mean)
    dyo_level.insert(4, 'Four hour moving mean (m)', dyo_four_hr_mean)
    dyo_level.insert(5, 'Six hour moving mean (m)', dyo_six_hr_mean)
    dyo_level.insert(6, 'Eight hour moving mean (m)', dyo_eight_hr_mean)
    dyo_level.insert(7, 'Twelve hour moving mean (m)', dyo_twelve_hr_mean)
    dyo_level.insert(8, 'Twenty four hour moving mean (m)', dyo_twentyfour_hr_mean)
    dyo_level.insert(9, 'Fourty eight hour moving mean (m)', dyo_fourtyeight_hr_mean)
    dyo_level.insert(10, 'Seventy two hour moving mean (m)', dyo_seventytwo_hr_mean)
    dyo_level.insert(11, 'Ninety six hour moving mean (m)', dyo_ninetysix_hr_mean)
    dyo_level.insert(12, 'One hundred and twenty hour moving mean (m)', dyo_onetwenty_hr_mean)

    # Add the numerical indexes to the rise/peak index data. Create a data frame out of the two series'
    # and sort the indexing out
    wff_level_peak_times = pd.DataFrame(wff_level['Water head[m]'][wff_level_peak_times_idx])
    wff_level_peak_times.insert(1, 'Sample number (5 min intervals)', wff_level_peak_times_idx)
    wff_temperature_peak_times = pd.DataFrame(wff_level['Temperature[°C]'][wff_temperature_peak_times_idx])
    wff_temperature_peak_times.insert(1, 'Sample number (5 min intervals)', wff_temperature_peak_times_idx)
    wff_temperature_trough_times = pd.DataFrame(wff_level['Temperature[°C]'][wff_temperature_trough_times_idx])
    wff_temperature_trough_times.insert(1, 'Sample number (5 min intervals)', wff_temperature_trough_times_idx)
    giedd_level_peak_times = pd.DataFrame(giedd_level['Water head[m]'][giedd_level_peak_times_idx])
    giedd_level_peak_times.insert(1, 'Sample number (5 min intervals)', giedd_level_peak_times_idx)
    giedd_temperature_peak_times = pd.DataFrame(giedd_level['Temperature[°C]'][giedd_temperature_peak_times_idx])
    giedd_temperature_peak_times.insert(1, 'Sample number (5 min intervals)', giedd_temperature_peak_times_idx)
    giedd_temperature_trough_times = pd.DataFrame(giedd_level['Temperature[°C]'][giedd_temperature_trough_times_idx])
    giedd_temperature_trough_times.insert(1, 'Sample number (5 min intervals)', giedd_temperature_trough_times_idx)
    dyo_level_peak_times = pd.DataFrame(dyo_level['Water head[m]'][dyo_level_peak_times_idx])
    dyo_level_peak_times.insert(1, 'Sample number (5 min intervals)', dyo_level_peak_times_idx)
    dyo_temperature_peak_times = pd.DataFrame(dyo_level['Temperature[°C]'][dyo_temperature_peak_times_idx])
    dyo_temperature_peak_times.insert(1, 'Sample number (5 min intervals)', dyo_temperature_peak_times_idx)
    dyo_temperature_trough_times = pd.DataFrame(dyo_level['Temperature[°C]'][dyo_temperature_trough_times_idx])
    dyo_temperature_trough_times.insert(1, 'Sample number (5 min intervals)', dyo_temperature_trough_times_idx)


    with pd.ExcelWriter('outputs/unified_logger_data_jul_2023_feb_2024.xlsx') as writer:
        nrw_tawe_level.to_excel(writer, sheet_name='nrw_tawe_level')
        nrw_dyo_rainfall.to_excel(writer, sheet_name='nrw_dyo_rainfall')
        wff_level.to_excel(writer, sheet_name='wff_level')
        wff_level_peak_times.to_excel(writer, sheet_name='wff_level_peak_times')
        wff_temperature_peak_times.to_excel(writer, sheet_name='wff_temperature_peak_times')
        wff_temperature_trough_times.to_excel(writer, sheet_name='wff_temperature_trough_times')
        giedd_level.to_excel(writer, sheet_name='giedd_level')
        giedd_level_peak_times.to_excel(writer, sheet_name='giedd_level_peak_times')
        giedd_temperature_peak_times.to_excel(writer, sheet_name='giedd_temperature_peak_times')
        giedd_temperature_trough_times.to_excel(writer, sheet_name='giedd_temperature_trough_times')
        dyo_level.to_excel(writer, sheet_name='dyo_level')
        dyo_level_peak_times.to_excel(writer, sheet_name='dyo_level_peak_times')
        dyo_temperature_peak_times.to_excel(writer, sheet_name='dyo_temperature_peak_times')
        dyo_temperature_trough_times.to_excel(writer, sheet_name='dyo_temperature_trough_times')


def main():
    # Name input files
    file1 = "CSV/VEI_DZ629_WFF_250209124536_DZ629.csv"
    file2 = "CSV/VEI_EW141_DYO_250209124546_EW141.csv"
    file3 = "CSV/NRW_DYO_20250208215247.csv"
    file4 = "CSV/NRW_TAWE_20250208215134.csv"

    ## Import data
    wff_level = import_logger_data(file1, 75, 33363)
    dyo_level = import_logger_data(file2, 51, 33363)
    nrw_dyo_rainfall = import_nrw_data(file3, 78, 11127)
    nrw_tawe_level = import_nrw_data(file4, 78, 11127)

    ## Identify peaks times
    dyo_level_peak_times_idx, dyo_temperature_peak_times_idx, dyo_temperature_trough_times_idx = identify_peaks(dyo_level)
    wff_level_peak_times_idx, wff_temperature_peak_times_idx, wff_temperature_trough_times_idx = identify_peaks(wff_level)

    ## Calculate the average level at the resurgence before each flood pulse - background levels
    (dyo_one_hr_mean, dyo_two_hr_mean, dyo_four_hr_mean, dyo_six_hr_mean, dyo_eight_hr_mean, dyo_twelve_hr_mean,
     dyo_twentyfour_hr_mean, dyo_fourtyeight_hr_mean, dyo_seventytwo_hr_mean, dyo_ninetysix_hr_mean,
     dyo_onetwenty_hr_mean) = calculate_averages(dyo_level)

    # Calculate stats and write to file
    calculate_stats('DYO', dyo_level)
    calculate_stats('WFF', wff_level)

    ## Plot all the data for comparison
    # plot_all_levels(nrw_tawe_level, nrw_dyo_rainfall, wff_level, dyo_level)

    # ## Plot each sink against rainfall & resurgence, and highlight peak/rise samples
    labels = ['WFF stage', 'NRW rainfall', 'DYO stage', 'DYO 120 hour moving mean', 'DYO water temperature']
    plot_details(dyo_level, dyo_level_peak_times_idx, dyo_temperature_peak_times_idx, wff_level,
                 wff_level_peak_times_idx, dyo_onetwenty_hr_mean, nrw_dyo_rainfall, labels)

    ## Create unified Excel output
    # write_excel(nrw_tawe_level,
    #                nrw_dyo_rainfall,
    #                wff_level,
    #                wff_level_peak_times_idx,
    #                wff_temperature_peak_times_idx,
    #                wff_temperature_trough_times_idx,
    #                giedd_level,
    #                giedd_level_peak_times_idx,
    #                giedd_temperature_peak_times_idx,
    #                giedd_temperature_trough_times_idx,
    #                dyo_level,
    #                dyo_level_peak_times_idx,
    #                dyo_temperature_peak_times_idx,
    #                dyo_temperature_trough_times_idx,
    #                dyo_one_hr_mean, dyo_two_hr_mean,
    #                dyo_four_hr_mean, dyo_six_hr_mean,
    #                dyo_eight_hr_mean, dyo_twelve_hr_mean, dyo_twentyfour_hr_mean,
    #                dyo_fourtyeight_hr_mean, dyo_seventytwo_hr_mean,
    #               dyo_ninetysix_hr_mean, dyo_onetwenty_hr_mean)


if __name__ == "__main__":
    main()