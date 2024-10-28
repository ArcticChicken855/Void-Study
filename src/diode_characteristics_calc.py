"""
This program is used to calculate the diode external resistance, saturation current, and ideality factor.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

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
    q = 1.602e-19  # Charge of electron
    k = 1.38e-23   # Boltzmann constant
    T = 300        # Temperature in Kelvin (room temp)
    return (n * k * T / q) * np.log((Id / Is) + 1) + R_ext * Id

def find_parameters(input_voltages, currents):
    """
    This is the function that performs the optimizaiton using least-squares.
    """
    initial_guess = [1E-12, 1.5, 0.25] # initial guess for Is, n, R_ext

    # set up bounds for the curve fit
    bounds = ([0, 1, 0], [np.inf, 2, np.inf]) #bounds for Is, n, R_ext

    params, covariance = curve_fit(diode_equation, currents, input_voltages, p0=initial_guess, bounds=bounds)

    # Extract the parameters
    Is, n, R_ext = params

    # Return the derived parameters
    return Is, n, R_ext

def main():
    project_dir = Path(__file__).parent.parent
    excel_file_path = project_dir / r'Non-code' / r'Void Study IV Curve.xlsx'

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
    Is, n, R_ext = find_parameters(input_voltages, currents)

    print(f"Is={Is:.3g}, n={n:.3g}, R_ext={R_ext:.3g}")
main()