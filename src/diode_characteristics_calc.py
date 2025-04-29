"""
This program is used to calculate the diode external resistance, saturation current, and ideality factor.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def normalize_units(df):
    """
    Normalize voltage and current values in a DataFrame to V and A.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Voltage' and 'Current' columns as strings.

    Returns:
    pd.DataFrame: DataFrame with normalized values as floats.
    """
    
    def convert_voltage(voltage_str):
        """Convert voltage string to float in volts (V)."""
        value, unit = voltage_str.split()
        value = float(value)
        if unit == 'mV':
            return value / 1000  # Convert millivolts to volts
        elif unit == 'V':
            return value  # Already in volts
        else:
            raise ValueError(f"Unknown voltage unit: {unit}")

    def convert_current(current_str):
        """Convert current string to float in amperes (A)."""
        value, unit = current_str.split()
        value = float(value)
        if unit == 'mA':
            return value / 1000  # Convert milliamperes to amperes
        elif unit == 'A':
            return value  # Already in amperes
        else:
            raise ValueError(f"Unknown current unit: {unit}")

    # Apply the conversion functions to the DataFrame
    df['Vf'] = df['Vf'].apply(convert_voltage)
    df['If'] = df['If'].apply(convert_current)

    return df

def diode_equation(Id, Is, n, R_ext):
    """
    This is a form of the diode equation that includes the external resistance.
    It takes in the diode current as the independent variable and returns the input voltage needed to achieve this current.
    It uses the saturation current Is, ideality factor n, and external resistance R_ext as model parameters.
    These parameters are the main optimizaiton objective.
    """
    q = 1          # Charge of electron in eV
    k = 8.61733E-5 # boltzmann constant in eV
    T = 300        # Temperature in Kelvin (room temp)

    Vd = (n * k * T / q) * np.log((Id / Is) + 1) + R_ext * Id
    return Vd

def Is_temp_dependence_eqn(T, a, r, p):
    """
    This is the equaiton for the temperature dependence of the saturation current.
    It has the form Is = AT^r * exp(-p*Eg/kT). https://www.sciencedirect.com/science/article/pii/S0924424715002137
    Eg also has a temp dependence given in [56] Y. P. Varshni, Physica 1967, 34, 149.
    The parameters A and r will be found using a curve fit on the Is data.
    """
    # first, get the value of Eg
    Eg0 = 1.1557 # eV
    alpha = 7.021E-4
    beta = 1108
    k = 8.61733E-5 # boltzmann constant in eV

    Eg = Eg0-alpha*T**2/(T+beta) # in eV

    # compute Is
    Is = a*T**r * np.exp(-p*Eg0/(k*T))
    return Is

def find_diode_parameters(input_voltages, currents):
    """
    This is the function that performs the optimizaiton using least-squares.
    It finds the diode parameters using the IV curve.
    """
    initial_guess = [96.514E-12, 1.5, 10E-3] # initial guess for Is, n, R_ext

    # set up bounds for the curve fit
    bounds = ([0, 1, 0], [1, 100, 0.5]) #bounds for Is, n, R_ext

    params, covariance = curve_fit(diode_equation, currents, input_voltages, p0=initial_guess, bounds=bounds, maxfev=1E10)

    # Extract the parameters
    Is, n, R_ext = params

    # Return the derived parameters
    return Is, n, R_ext

def find_Is_params(currents, temps):
    """
    This function finds the Is temp dependence bu curve-fitting to experimental data.
    """
    initial_guess = [312, 3, 1] # initial guess for A, r, p

    # set up bounds for the curve fit
    bounds = ([0, -1, 0], [np.inf, 10, 100]) #bounds for Is, A, r, p

    params, covariance = curve_fit(Is_temp_dependence_eqn, temps, currents, p0=initial_guess, bounds=bounds, maxfev=1E10)

    # Extract the parameters
    a, r, p = params

    # Return the derived parameters
    return a, r, p

def plot_Is(currents, temps, a, r, p):
    """
    Make a plot of Is over T^-1 for both experimental data and the curve-fit stuff.
    """
    c_temps = np.linspace(min(temps), max(temps), 100000)
    c_currents = np.zeros((100000))

    for i, temp in enumerate(c_temps):
        c_currents[i] = Is_temp_dependence_eqn(temp, a, r, p)

    fig, ax = plt.subplots()

    ax.scatter(1/temps, currents, label='Experimental', color='blue')
    ax.plot(1/c_temps, c_currents, label='Curve-fit', color='red')

    ax.set_yscale('log')
    ax.set_xlabel('T^-1')
    ax.set_ylabel('Is (uA)')
    ax.set_title('Is over temp^-1')
    plt.legend()

    plt.show()

def plot_IV_curve(currents, voltages, Is, n, R_ext):
    """
    Make a plot of the IV curve at room temp.
    """
    c_currents = np.linspace(min(currents), max(currents), 100000)
    c_voltages = np.zeros((100000))

    for i, current in enumerate(c_currents):
        c_voltages[i] = diode_equation(current, Is, n, R_ext)

    fig, ax = plt.subplots()

    ax.scatter(voltages, currents, label='Experimental', color='blue')
    ax.semilogy(c_voltages, c_currents, label='Curve-fit', color='red')

    ax.set_xlabel('Vd (V)')
    ax.set_ylabel('Id (A)')
    ax.set_title('Diode IV curve at room temp')
    plt.legend()

    plt.show()
    

def main():
    project_dir = Path(__file__).parent.parent
    excel_file_path = project_dir / r'Experimental Data' / r'Void Study IV Curve.xlsx'
    Is_file_path = project_dir / r'Experimental Data' / r'Diode Is measurement.xlsx'

    df = pd.read_excel(excel_file_path)
    
    # Delete the first column, it is just the useless index
    df = df.drop(df.columns[0], axis=1)

    # cut out the first 10 values, they are not accurate
    df = df.drop(df.index[:10])
    
    # normalize all entries to be in V or A, no mV or mA. Also makes the datatype into a float.
    df = normalize_units(df)
    
    # convert to numpy :)
    input_voltages = df['Vf'].to_numpy()
    currents = df['If'].to_numpy()

    # Perform optimization
    Is, n, R_ext = find_diode_parameters(input_voltages, currents)

    # look at the Is measurements at 65.8V
    df = pd.read_excel(Is_file_path)
    df = df.drop(df.columns[5:], axis=1)

    # convert to numpy
    reverse_currents = df['Reverse Current (uA)'].to_numpy()
    temperatures = df['Temperature (K)'].to_numpy()

    # find Is temp dependence params
    a, r, p = find_Is_params(reverse_currents, temperatures)

    # make plots
    plot_Is(reverse_currents, temperatures, a, r, p)
    plot_IV_curve(currents, input_voltages, Is, n, R_ext)

    print(f'A={a:.3g}, r={r:.3g}, p={p:.3g}')
    print(f"Is={Is:.3g}, n={n:.3g}, R_ext={R_ext:.3g}")

    pwr = input_voltages * currents
    engy = 0
    for power in pwr:
        engy += power * 150E-6

    print(f"Total energy: {engy}")
main()

"""
Why this doesn't work:
In a diode, there are actually four separate currents that contribute to the overall forward current: 
Low level depletion region recombination Ir
Low-level injection Il
High-level injection Ih
Emitter recombination currents Ih
The relevant ones for a power diode IV curve are Il and Ih.
At low forward currents, the injected minority carrier concentration is less than the majority carrier concentration, which gives Il.
At high forward currents, the injected minority carrier concentration is equal to or greater than the majority carrier concentration, which gies Ih.
As you approach this point, the IV curve deviates from the expected exponential, the curve bends down as seen in the IV curve data.
"""