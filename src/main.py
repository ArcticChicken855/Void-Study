from pathlib import Path
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit

import data_formatting

def calculate_figure_dimensions(num_graphs):
    """
    Find the number of rows and columns necessary to fit all of the graphs.
    Try to make it as close to an even square as possible, if not, increase the
    num_columns before num_rows
    """

    square_length = np.ceil(np.sqrt(num_graphs))
    if (square_length * (square_length - 1)) >= num_graphs:
        num_columns = square_length
        num_rows = square_length - 1
    else:
        num_columns = square_length
        num_rows = square_length

    return int(num_columns), int(num_rows)

def detect_peaks(tau_intensity_data, label, current, neighbors_to_beat=2, prominence=None):
    """
    Uses scipy.signal.find_peaks() to find the peaks in the tau intensity
    data. Returns a list of indices where there are peaks.
    """

    intensities_to_test = tau_intensity_data[label][current].values
    indices_of_peaks, peak_data = find_peaks(intensities_to_test, distance=int(neighbors_to_beat/2), prominence=prominence)
    peak_prominences = peak_data['prominences']
    return indices_of_peaks, peak_prominences

def plot_tau_intensity(ax, tau_time_axis, tau_intensity_data, label, current, peak_finder_settings, show_peaks=True, label_peaks=True):
    """
    Plot the tau intensity for one test, along with the detected peaks.
    Returns the matplotlib figure.
    """
    # check to see if the data is valid
    if label not in tau_intensity_data:
        raise ValueError(f"The specified label \'{label}\' was not detected in the dataset")
    else:
        if current not in tau_intensity_data[label]:
            raise ValueError(f"The specified current \'{current}\' was not detected in the dataset for the label \'{label}\'")
        
    # get the peak indices, and create a list of peaks from them
    if show_peaks is True:
        peak_indices, *_ = detect_peaks(tau_intensity_data, label, current, peak_finder_settings['neighbors_to_beat'], peak_finder_settings['prominence'])

        peak_times = []
        peak_intensities = []
        for i in peak_indices:
            peak_time = float(tau_time_axis.iloc[:, 0].tolist()[i])
            peak_intensity = tau_intensity_data[label][current].tolist()[i]
            peak_times.append(peak_time)
            peak_intensities.append(peak_intensity)
            
    # plot the data
    ax.plot(tau_time_axis, tau_intensity_data[label][current])
    if show_peaks is True:
        ax.scatter(peak_times, peak_intensities, color='red')

    # Add labels to the peaks
    if label_peaks is True:
        for i, time in enumerate(peak_times):
            point = (time, peak_intensities[i])
            ax.scatter([point[0]], [point[1]], color='red')
            
            text_pos = [5, 5]
            if time < 0.1:
                text_pos[0] = -30
            if peak_intensities[i] == max(tau_intensity_data[label][current].values):
                text_pos = [10, -5]
            ax.annotate(f't={point[0]:.2e}',
                        xy=point, xytext=text_pos, textcoords='offset points', fontsize = 8)

    # use logarithmic x-scaling
    ax.set_xscale('log')

    # set the tick marks

    # set the title and axis labels
    ax.set_title(f'Tau Intensity, {label}, {current}', fontsize=12)
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel("Tau Intensity [K / W / -]", fontsize=10)

    return ax

def plot_tau_peaks(ax, tau_time_axis, tau_intensity_data, peak_finder_settings, xlimits=None, ylimits=None, labels='all', currents='all', scale_by_prominence=True):
    """
    Make a stem plot of all of the detected peaks, 
    optionally scaled by their prominence factor.
    """

    # if labels='all', make a list of labels to loop over
    if labels == 'all':
        labels = list(tau_intensity_data.keys())

    # loop over the dataset, add each prominence to a sum, and then plot the sum
    prominence_sum = np.zeros((len(tau_time_axis)), dtype=float)

    for label in labels:
        # check to see if label exists
        if label not in tau_intensity_data.keys():
            raise ValueError(f"The specified label \'{label}\' was not detected in the dataset")

        # if currents == 'all', loop over all of the currents under the label
        if currents == 'all':
            currents_to_test = list(tau_intensity_data[label].columns)
        else:
            currents_to_test = currents

        for current in currents_to_test:
            # check to see if current exists
            if current not in tau_intensity_data[label].columns:
                raise ValueError(f"The specified current \'{current}\' was not detected in the dataset for the label \'{label}\'")
            
            # calculate the peaks, along with the prominences
            peak_indices, peak_prominences = detect_peaks(tau_intensity_data, label, current, peak_finder_settings['neighbors_to_beat'], peak_finder_settings['prominence'])
            
            # add the prominences to the running sum
            for peak_index, absolute_index in enumerate(peak_indices):
                if scale_by_prominence is True:
                    prominence_sum[absolute_index] += peak_prominences[peak_index]
                else:
                    prominence_sum[absolute_index] += 1

    # plot the data
    markerline, stemlines, baseline = ax.stem(tau_time_axis.iloc[:, 0], prominence_sum)
    #plt.setp(markerline, 'markersize', 0)
    #plt.setp(stemlines, 'linewidth', 3)

    # use logarithmic x-scaling
    ax.set_xscale('log')

    if scale_by_prominence is True:
        #ax.set_yscale('log')
        pass

    # set axis limits
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)

    # set the title and axis labels
    if scale_by_prominence is True:
        ax.set_title(f'Tau peaks scaled by prominence vs time')
        ax.set_ylabel("Sum of prominence")
    else:
        ax.set_title(f'Number of Tau peaks vs time')
        ax.set_ylabel("# of peaks")

    ax.set_xlabel('Time [s]')

    return ax

def analytical_void_equation(v, alpha, beta, r_naught):
    """
    Enter v as a number from 0 to 100 (void percent)
    """
    v_r = v / 100
    return r_naught + beta * (1 - (1 + alpha)*v_r) / ((1 - v_r) ** 2)

def plot_tau_vs_voids(ax, tau_time_axis, tau_intensity_data, void_data, tau_range, current, peak_finder_settings, invert_tau=False, labels='all', trendline='linear'):
    """
    Generate a plot of the Tau time vs voids at a specific current. The tau is 
    determined to be the most prominent peak within a given time range. If there is 
    no peak within that range, it will return a warning and plot it with the datapoint
    left out.
    """
    # make ordered lists for voids and tau time
    void_list = []
    tau_list = []

    # if labels='all', make a list of labels to loop over
    if labels == 'all':
        labels = list(tau_intensity_data.keys())

    for label in labels:
        # check to see if label exists
        if label not in tau_intensity_data.keys():
            raise ValueError(f"The specified label \'{label}\' was not detected in the dataset")

        # check to see if current exists
        if current not in tau_intensity_data[label].columns:
            raise ValueError(f"The specified current \'{current}\' was not detected in the dataset for the label \'{label}\'")
            
        # calculate the peaks, along with the prominences
        peak_indices, peak_prominences = detect_peaks(tau_intensity_data, label, current, peak_finder_settings['neighbors_to_beat'], peak_finder_settings['prominence'])
        
        # check to see if there are any peaks within the range
        highest_prominence = 0
        time_of_highest_peak = 0
        for peak_index, absolute_index in enumerate(peak_indices):
            if tau_range[0] <= tau_time_axis.iloc[absolute_index, 0] <= tau_range[1]:
                if peak_prominences[peak_index] > highest_prominence:
                    highest_prominence = peak_prominences[peak_index]
                    time_of_highest_peak = tau_time_axis.iloc[absolute_index, 0]

            
        if time_of_highest_peak == 0:
            print(f'Warning: there are no peaks for {label},{current} within the specified tau range')
        else:
            void_list.append(void_data[label])
            tau_list.append(time_of_highest_peak)

    # if desired, invert the tau axis
    if invert_tau is True:
        y_data = []
        for tau in tau_list:
            y_data.append(1/tau)
    else:
        y_data = tau_list

    # generate the trendline
    if trendline is not None:
        if trendline == 'linear':
            coefficients = np.polyfit(void_list, y_data, 1)
            linfunc = np.poly1d(coefficients)
            y_fit = linfunc(void_list)

        elif trendline == 'analytical':
            optimal_parameters, param_cov = curve_fit(analytical_void_equation, void_list, y_data, bounds=[(0, 0, 0), (1, 1E3, 10)], method='trf', maxfev=100000)
            alpha = optimal_parameters[0]
            beta = optimal_parameters[1]
            r_naught = optimal_parameters[2]
            print(optimal_parameters)
            y_fit = []
            for void in void_list:
                y_fit.append(analytical_void_equation(void, alpha, beta, r_naught))

        # find the R^2 value
        correlation_matrix = np.corrcoef(y_data, y_fit)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy**2

    # plot it
    ax.scatter(void_list, y_data)

    if trendline is not None:
        ax.plot(sorted(void_list), sorted(y_fit), color=(0.3, 0.7, 1.0), linewidth=0.8, linestyle='--')
        plt.text(min(void_list) + 0.4*(max(void_list)-min(void_list)), min(y_fit) + 1.0*(max(y_fit) - min(y_fit)), f'$R^2 = {r_squared:.3f}$', fontsize=12)

    if invert_tau is True:
        ax.set_title(f'Inversion of Most Prominent Tau peak within ({tau_range[0]:.3g},{tau_range[1]:.3g}) vs Voids at {current}')
        ax.set_ylabel('Inverted Time of Tau Peak (s^-1)')
        ax.set_xlabel('Void percentage')
    else:
        ax.set_title(f'Most Prominent Tau peak within ({tau_range[0]:.3g},{tau_range[1]:.3g}) vs Voids at {current}')
        ax.set_ylabel('Time of Tau Peak (s)')
        ax.set_xlabel('Void percentage')

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))

    if trendline is None:
        return ax
    else:
        return ax, r_squared

def plot_zth_vs_voids(ax, zth_time_axis, zth_data, void_data, specified_time, current, labels='all', trendline='linear', invert_zth=False, plot=True):
    """
    Make a plot of the voids vs the zth at a specific time and current
    """
    # find the index of the time closest to the specified time
    best_distance = abs(zth_time_axis.iloc[0, 0] - specified_time)
    index_of_best = 0
    best_time = zth_time_axis.iloc[0, 0]
    for i, time in enumerate(zth_time_axis.iloc[:, 0]):
        dist = abs(time - specified_time)
        if dist < best_distance:
            best_distance = dist
            index_of_best = i
            best_time = time

    # make ordered lists for voids and tau time
    void_list = []
    zth_list = []

    # if labels='all', make a list of labels to loop over
    if labels == 'all':
        labels = list(zth_data.keys())

    for label in labels:
        # check to see if label exists
        if label not in zth_data.keys():
            raise ValueError(f"The specified label \'{label}\' was not detected in the dataset")

        # check to see if current exists
        if current not in zth_data[label].columns:
            raise ValueError(f"The specified current \'{current}\' was not detected in the dataset for the label \'{label}\'")
            
        # add the zth into the list along with the voids
        zth_list.append(zth_data[label][current].iloc[index_of_best])
        void_list.append(void_data[label])

    # invert zth into admittance if desired
    if invert_zth is True:
        y_data = []
        for zth in zth_list:
            y_data.append(1/zth)
    else:
        y_data = zth_list

    # generate the trendline
    if trendline is not None:
        if trendline == 'linear':
            coefficients = np.polyfit(void_list, y_data, 1)
            linfunc = np.poly1d(coefficients)
            y_fit = linfunc(void_list)

        elif trendline == 'analytical':
            optimal_parameters, param_cov = curve_fit(analytical_void_equation, void_list, y_data, bounds=[(0, 0, 0), (1, 1E3, 10)], method='trf', maxfev=100000)
            alpha = optimal_parameters[0]
            beta = optimal_parameters[1]
            r_naught = optimal_parameters[2]
            print(optimal_parameters)
            y_fit = []
            for void in void_list:
                y_fit.append(analytical_void_equation(void, alpha, beta, r_naught))

        # find the R^2 value
        correlation_matrix = np.corrcoef(y_data, y_fit)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy**2

    # plot it
    if plot is True:
        ax.scatter(void_list, y_data)

        if trendline is not None:
            ax.plot(sorted(void_list), sorted(y_fit), color=(0.3, 0.7, 1.0), linewidth=0.8, linestyle='--')
            plt.text(min(void_list) + 0.4*(max(void_list) - min(void_list)), min(y_data) + 0.9*(max(y_data)-min(y_data)), f'$R^2 = {r_squared:.3f}$', fontsize=12)

        if invert_zth is True:
            ax.set_title(f'Yth vs Voids at t={best_time:.3f} (s) and {current}')
            ax.set_ylabel('Yth [W / K]')
            ax.set_xlabel('Void percentage')
        else:
            ax.set_title(f'Zth vs Voids at t={best_time:.3f} (s) and {current}')
            ax.set_ylabel('Zth [K / W]')
            ax.set_xlabel('Void percentage')

        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))

    if trendline is None:
        return ax
    else:
        return ax, r_squared

def plot_void_zth_r_squared_vs_time(ax, zth_time_axis, zth_data, void_data, current, labels='all', trendline='linear', invert_zth=False):
    """
    For each time step, calculate the r^2 val for the void-zth relationship, and
    then plot this quantity over time
    """
    r_squared_list = []
    for time in zth_time_axis.iloc[:, 0]:
        *_, r_squared = plot_zth_vs_voids(ax, zth_time_axis, zth_data, void_data, time, current, labels, trendline, invert_zth=invert_zth, plot=False)
        r_squared_list.append(r_squared)
        
    # now plot it
    ax.plot(zth_time_axis.iloc[:, 0], r_squared_list)

    if invert_zth is True:
        ax.set_title(f'R^2 of void-Yth relationship over time at {current}')
    else:
        ax.set_title(f'R^2 of void-Zth relationship over time at {current}')

    ax.set_xscale('log')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f'R^2')

    return ax

def plot_zth_vs_power(ax, zth_time_axis, zth_data, power_step_data, specified_time, currents='all', labels='all', print_percents=True):
    """
    Plot the zth at a specified time versus the power. Color each label
    differently and make a legend. Also, make connecting lines for each
    label.
    """
    # find the index of the time closest to the specified time
    best_distance = abs(zth_time_axis.iloc[0, 0] - specified_time)
    index_of_best = 0
    best_time = zth_time_axis.iloc[0, 0]
    for i, time in enumerate(zth_time_axis.iloc[:, 0]):
        dist = abs(time - specified_time)
        if dist < best_distance:
            best_distance = dist
            index_of_best = i
            best_time = time

    # if labels='all', make a list of labels to loop over
    if labels == 'all':
        labels = list(zth_data.keys())    

    # Define a color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    color_idx = 0

    for label in labels:
        # check to see if label exists
        if label not in zth_data.keys():
            raise ValueError(f"The specified label \'{label}\' was not detected in the dataset")

        # if currents == 'all', loop over all of the currents under the label
        if currents == 'all':
            currents_to_test = list(zth_data[label].columns)
        else:
            currents_to_test = currents

        zth = []
        pwr = []
        for current in currents_to_test:

            # check to see if current exists
            if current not in zth_data[label].columns:
                raise ValueError(f"The specified current \'{current}\' was not detected in the dataset for the label \'{label}\'")
            
            zth.append(zth_data[label][current].iloc[index_of_best])
            pwr.append(power_step_data[label][current])

        ax.plot(pwr, zth, marker='o', linestyle='-', color=colors[color_idx], label=f'{label}')
        color_idx += 1

        if print_percents is True:
            zth = np.array(zth)
            avg = np.mean(zth)
            differences = np.abs(zth - avg)
            percentage_change = max(differences) / avg * 100
            print(f't={best_time:.3g},{label}:{percentage_change:.3g}%')
    
    ax.legend()
    ax.set_xlabel('Power [W]')
    ax.set_ylabel('Zth [K / W]')
    ax.set_title(f'Zth vs Power at t={best_time:.2g}')

    return ax


def main(excel_file_path, project_name_in_power_tester, plots_to_show):
    # get the excel file opened
    excel_sheets = pd.read_excel(excel_file_path, sheet_name=None)

    # get the formatted data out of the excel file
    zth_formatted = data_formatting.rename_columns(excel_sheets['Zth'], project_name_in_power_tester)
    tau_formatted = data_formatting.rename_columns(excel_sheets['Tau Intensity'], project_name_in_power_tester)

    tau_time_axis, tau_intensity_data = data_formatting.format_timed_data(tau_formatted)
    zth_time_axis, zth_data =           data_formatting.format_timed_data(zth_formatted)

    void_data = excel_sheets['Void Data'].iloc[0].to_dict()
    power_step_data = data_formatting.format_power_step(excel_sheets['Power Step'])

    # initialize settings for peak finder
    peak_finder_settings = dict()
    peak_finder_settings['neighbors_to_beat'] = round(len(tau_time_axis) / 10)
    peak_finder_settings['prominence'] = 0.003

    # plot the tau intensity data as well as the peaks
    if ("Tau Intensity" in plots_to_show) or ('all' == plots_to_show):
    
        labels_to_plot = ['L3']
        currents_to_plot = ['24A']

        # calculate the number of graphs needed
        num_graphs = 0

        # if labels='all', make a list of labels to loop over
        if labels_to_plot == 'all':
            labels_to_plot = list(tau_intensity_data.keys())

        for label in labels_to_plot:
            if currents_to_plot == 'all':
                num_graphs += len(tau_intensity_data[label].columns)
            else:
                num_graphs += len(currents_to_plot)

        # get num_columns and num_rows for the figure
        num_columns, num_rows = calculate_figure_dimensions(num_graphs)

        # make the figure and flatten axes
        tau_intensity_fig, axes = plt.subplots(num_rows, num_columns)
        if num_graphs > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        ax_idx = 0

        # plot the figures on the axis
        for label in labels_to_plot:
            if currents_to_plot == 'all':
                currents = list(tau_intensity_data[label].columns)
            else:
                currents = currents_to_plot
            
            for current in currents:
                axes[ax_idx] = plot_tau_intensity(axes[ax_idx], tau_time_axis, tau_intensity_data, label, current, peak_finder_settings)
                ax_idx += 1

        tau_intensity_fig.suptitle("Tau Intensity", fontsize=16)
        tau_intensity_fig.subplots_adjust(hspace=0.5, wspace=0.4)

    # plot a stem plot of tau intensities over time
    if ("Tau Peaks" in plots_to_show) or ('all' == plots_to_show):
        
        num_graphs = 6

        # get num_columns and num_rows for the figure
        num_columns, num_rows = calculate_figure_dimensions(num_graphs)

        # make the figure and flatten axes
        tau_peak_fig, axes = plt.subplots(num_rows, num_columns)
        if num_graphs > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        ax_idx = 0

        # make the plots
        xlim = (min(zth_time_axis.iloc[:, 0]), 1E-2)
        ylim1 = (0, 0.25)
        ylim2 = (0, 10)

        """
        axes[ax_idx] = plot_tau_peaks(axes[ax_idx], tau_time_axis, tau_intensity_data, peak_finder_settings, xlimits=xlim, ylimits=ylim1, labels='all', currents='all', scale_by_prominence=True)
        ax_idx += 1

        axes[ax_idx] = plot_tau_peaks(axes[ax_idx], tau_time_axis, tau_intensity_data, peak_finder_settings, xlimits=xlim, ylimits=ylim2, labels='all', currents='all', scale_by_prominence=False)
        ax_idx += 1


        currents = ['10A', '15A', '20A', '22A', '24A']
        axes[ax_idx] = plot_tau_peaks(axes[ax_idx], tau_time_axis, tau_intensity_data, peak_finder_settings, xlimits=xlim, ylimits=ylim1, labels='all', currents=currents, scale_by_prominence=True)
        ax_idx += 1

        axes[ax_idx] = plot_tau_peaks(axes[ax_idx], tau_time_axis, tau_intensity_data, peak_finder_settings, xlimits=xlim, ylimits=ylim2, labels='all', currents=currents, scale_by_prominence=False)
        ax_idx += 1
        """

        for c in ['5A', '10A', '15A', '20A', '22A', '24A']:
            if c == '5A':
                labels = ['C4', 'C3', 'C2', 'C1', 'L5', 'L4', 'L3', 'L2', 'L1']
            else:
                labels = 'all'
            ylim3 = (0, 0.2)
            axes[ax_idx] = plot_tau_peaks(axes[ax_idx], tau_time_axis, tau_intensity_data, peak_finder_settings, xlimits=xlim, ylimits=ylim3, labels=labels, currents=[c], scale_by_prominence=True)
            ax_idx += 1
        
    # plot voids vs tau time
    if ("Tau vs Voids" in plots_to_show) or ('all' == plots_to_show):

        void_tau_fig, axes = plt.subplots(1, 1)
        tau_range = (0.7E-3, 4E-3)
        current = '24A'
        labels = ['C5', 'C4', 'C3', 'C2', 'C1']
        labels = ['L5', 'L4', 'L3', 'L2', 'L1']
        axes, *_ = plot_tau_vs_voids(axes, tau_time_axis, tau_intensity_data, void_data, tau_range, current, peak_finder_settings, labels=labels, trendline='analytical', invert_tau=False)

    # plot voids vs zth at a specific time
    if ("Zth vs Voids" in plots_to_show) or ('all' == plots_to_show):

        specified_time = 2E-3
        current = '22A'
        void_zth_fig, axes = plt.subplots(1, 1)
        labels = ['C5', 'C4', 'C3', 'C2', 'C1']
        labels = ['L5', 'L4', 'L3', 'L2', 'L1']
        axes, *_ = plot_zth_vs_voids(axes, zth_time_axis, zth_data, void_data, specified_time, current, labels=labels, trendline='analytical')

    # plot void-zth r^2 value over time
    if ("Zth-void r-squared" in plots_to_show) or ('all' == plots_to_show):

        void_zth_r_squared_fig, axes = plt.subplots(1, 1)
        current = '24A'

        axes = plot_void_zth_r_squared_vs_time(axes, zth_time_axis, zth_data, void_data, current)

        void_zth_r_squared_fig2, axes2 = plt.subplots(1, 1)
        current = '24A'

        axes2 = plot_void_zth_r_squared_vs_time(axes2, zth_time_axis, zth_data, void_data, current, invert_zth=True)

    if ("Zth vs Power" in plots_to_show) or ('all' == plots_to_show):
        labels_to_plot = 'all'
        currents_to_plot = 'all'
        times_to_plot = [0.0001, 0.001, 0.1, 1, 10, 100]

        # calculate the number of graphs needed
        num_graphs = len(times_to_plot)

        # if labels='all', make a list of labels to loop over
        if labels_to_plot == 'all':
            labels_to_plot = list(zth_data.keys())

        # get num_columns and num_rows for the figure
        num_columns, num_rows = calculate_figure_dimensions(num_graphs)

        # make the figure and flatten axes
        fig, axes = plt.subplots(num_rows, num_columns)
        if num_graphs > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        ax_idx = 0

        # plot the figures on the axis
        for t in times_to_plot:
            axes[ax_idx] = plot_zth_vs_power(axes[ax_idx], zth_time_axis, zth_data, power_step_data, t, currents_to_plot, labels_to_plot)
            ax_idx += 1

        fig.suptitle("Heebee Jeebee", fontsize=16)
        fig.subplots_adjust(hspace=0.5, wspace=0.4)

    plt.show()
    return


# run command
script_dir = Path(__file__).parent
excel_file_path = script_dir.parent / 'Experimental Data' / 'Void Study FULL DOC.xlsx'
project_name_in_power_tester = "NAHANS VOID STUDY"
main(excel_file_path, project_name_in_power_tester, plots_to_show=['Zth vs Power'])

# add physical fit