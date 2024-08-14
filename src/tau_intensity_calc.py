import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit
from scipy.signal import lombscargle
import scipy.ndimage

import data_formatting

def df_to_numpy(df):
    """
    Takes in a 2D dataframe and converts it to a dictionary of numpy arrays, where the key is the column title.
    """
    numpy_dict = dict()
    for column in df.columns:
        numpy_dict[column] = df[column].to_numpy(dtype=np.float64)

    return numpy_dict

def differentiate(f, time): #returns an array of size-1 of derivatives as well as the times

    df = np.zeros(len(f)-1)
    dtime = np.zeros(len(time)-1)

    for i, val in enumerate(f):
        if i == 0:
            pass
        else:
            dt = time[i] - time[i-1]
            df[i-1] = (f[i] - f[i-1]) / dt
            dtime[i-1] = time[i-1] + 0.5*dt

    return dtime, df

def remove_zero_times(temp, time):
    """
    Just removes data entries if time=0.
    """
    nonzero_indeces = np.nonzero(time)[0]
    return time[nonzero_indeces], temp[nonzero_indeces]

def extrapolate_root_t(temp, time, cutoff_time, max_extrapolation_time):
    """
    This returns new time and temp arrays with the values before the cutoff time replaced with an extrapolation that assumes temp is linear with sqrt(t).
    The extrapolation considers values from the 
    """
    # find the index of the time closest to the cutoff time
    min_idx = np.abs(time - cutoff_time).argmin()
    max_idx = np.abs(time - max_extrapolation_time).argmin()

    # get the subsection of data between the bounds
    temp_sc = temp[min_idx:max_idx]
    time_sc = time[min_idx:max_idx]

    # get the root time vector
    root_time = np.zeros(len(time_sc), dtype='float')
    for i, t in enumerate(time_sc):
        root_time[i] = np.sqrt(t)

    # find the line of best fit
    coefficients = np.polyfit(root_time, temp_sc, 1)
    linfunc = np.poly1d(coefficients)

    # make the vector to return
    extrp_temp = np.zeros(len(temp))

    # replace vals in "temp" with extrapolated values if below the cutoff time
    for i, temperature in enumerate(temp):
        if time[i] <= cutoff_time:
            extrp_temp[i] = linfunc(np.sqrt(time[i]))
        else:
            extrp_temp[i] = temperature

    return extrp_temp

def compute_H(times, temps, nfreqs, window='none'):
    """
    This will get |H(jw)| of the thermal system.
    """
    dtime, dtemp = differentiate(temps, times)

    h = np.flip(dtemp)
    h_time = dtime

    if window == 'none':
        h_windowed = h
    elif window == 'blackman':
        h_windowed = h * np.blackman(len(h))

    H = np.fft.fft(h_windowed)
    freqs = np.fft.fftfreq(len(h_windowed), d=(times[1]-times[0]))

    # only want the positive freqs (not zero)
    H = H[1:(len(H) // 2)]
    freqs = freqs[1:(len(freqs) // 2)]

    # now, try dividing by 1-e^-jwD
    delay = 1
    exp_term = 0 + 1j*freqs*2*np.pi

    H = H / (1 - np.exp(exp_term))

    return freqs, H


def main(excel_file_path, project_name_in_power_tester):
    # get the excel file opened
    excel_sheets = pd.read_excel(excel_file_path, sheet_name=None)

    # get the formatted data out of the excel file
    formatted = data_formatting.rename_columns(excel_sheets['Linear'], project_name_in_power_tester)

    times_df, temps_df = data_formatting.format_timed_data(formatted)

    # turn the stuff into numpy arrays while removing values where t==0
    raw_times = df_to_numpy(times_df)['Time [s]']

    temps = dict()
    for label in temps_df.keys():
        temps[label] = dict()
        raw_temps = df_to_numpy(temps_df[label])
        for current in raw_temps.keys():
            times, temps[label][current] = remove_zero_times(raw_temps[current], raw_times)

    # extrapolate back to t=0 using the method that temp is linear with sqrt(t).
    extrapolation_cutoff_time = 200E-6
    max_extrapolation_time = 400E-6

    for label in temps.keys():
        for current in temps[label].keys():
            temps[label][current] = extrapolate_root_t(temps[label][current], times, extrapolation_cutoff_time, max_extrapolation_time)

    # interpolate the data such that each point has consistent time spacing, and also set a max time
    N = int(1E5 - 1)
    max_time = max(times)

    s_times = np.linspace(times[0], max_time, N)
    s_temps = temps

    for label in temps.keys():
        for current in temps[label].keys():
            s_temps[label][current] = np.interp(s_times, times, temps[label][current])

    # initialize settings for peak finder
    peak_finder_settings = dict()
    peak_finder_settings['neighbors_to_beat'] = round(len(s_times) / 10)
    peak_finder_settings['prominence'] = 0.003

    # now run the fourier transform
    label_to_test = 'L4'
    current_to_test = '24A'
    temps_to_test = s_temps[label_to_test][current_to_test]

    freqs, H = compute_H(s_times, temps_to_test, 1E5, window='none')

    # filter H
    abs_H_filtered = scipy.ndimage.convolve1d(np.abs(H), np.ones(len(H) // 100), axis=0)

    df, dH = differentiate(abs(H), freqs)
    d2f, d2H = differentiate(dH, df)

    d2Hf = scipy.ndimage.convolve1d(d2H, np.ones(len(d2H) // 5), axis=0)

    taus = 1/freqs

    fig, axes = plt.subplots(3, 1)
    axes = axes.flatten()
    ax_idx = 0

    axes[ax_idx].plot(freqs, abs_H_filtered)
    axes[ax_idx].set_xscale('log')
    axes[ax_idx].set_yscale('log')
    axes[ax_idx].set_title('H vs f')
    axes[ax_idx].set_xlabel('Frequency (Hz)')
    ax_idx += 1

    axes[ax_idx].plot(taus, abs_H_filtered)
    #axes[ax_idx].set_xscale('log')
    #axes[ax_idx].set_yscale('log')
    axes[ax_idx].set_title('H vs tau')
    axes[ax_idx].set_xlabel('Time (s)')
    axes[ax_idx].set_xlim([min(taus), 1E-2])
    #axes[ax_idx].set_ylim([0, abs_H_filtered[np.abs(taus - 1E-2).argmin()]])
    ax_idx += 1

    axes[ax_idx].stem(d2f, d2H)
    axes[ax_idx].set_xscale('log')
    axes[ax_idx].set_yscale('log')
    axes[ax_idx].set_title('d2H vs f')
    axes[ax_idx].set_xlabel('Frequency (Hz)')
    ax_idx += 1

    plt.show()
    return


# run command
excel_file_path = excel_file_path = "C:\\Users\\natha\\Alabama\\baker Research\\Void Study\\Experimental Data\\Raw measured response.xlsx"
project_name_in_power_tester = "NAHANS VOID STUDY"
main(excel_file_path, project_name_in_power_tester)

# add physical fit