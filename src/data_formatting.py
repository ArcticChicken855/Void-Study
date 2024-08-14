# this module is for taking the data from an excel sheet and entering it
# into a dataframe, with the correct formatting

import pandas as pd

def extract_test_conditions_from_string(default_string, project_name_in_power_tester):
    """
    given the default string, return a string with the following format:
    label,current
    """
    project_name_formatted = project_name_in_power_tester.replace('_', ' ')
    project_name_length = len(project_name_formatted.split())

    spaced_str = default_string.replace('_', ' ')
    split_str = spaced_str.split()
    current = split_str[3 + project_name_length]
    position = split_str[7 + project_name_length]

    if position == "pos1":
        label = split_str[4 + project_name_length]
    elif position == "pos2":
        label = split_str[5 + project_name_length]
    else:
        raise NameError("Position is invalid")
    
    formatted_string = f'{label},{current}'
    return formatted_string

def rename_columns(dataframe_to_rename, project_name_in_power_tester):
    """
    Look through the column names, wherever the column title begins with
    <string_key>, replace with the extracted test conditions
    """
    new_column_names = []
    for column_name in dataframe_to_rename.columns:
        if column_name.startswith(project_name_in_power_tester):
            test_conditions = extract_test_conditions_from_string(column_name, project_name_in_power_tester)
            new_column_names.append(test_conditions)
        else:
            new_column_names.append('NoName')

    dataframe_to_rename.columns = new_column_names        
    return dataframe_to_rename

def format_timed_data(input_dataframe):
    """
        Take in a single dataframe corresponding to an excel sheet,
        return a few things. First, return a 1-column dataframe
        corresponding to the time of the data. Then, return a dictionary
        of dataframes where the label is the key and the value is a table
        of the data at each current level tested.

        - Assuming that the colums are [time][value][time][value]...
        - Assuming that the first row is the unit
    """

    # make the dataframe for the time
    time_axis = pd.DataFrame(input_dataframe.iloc[1:, 0])
    time_axis.columns = ['Time [s]']
    
    # loop over the input columns that have a name. If you come across
    # a name that is not yet in the dictionary, add a new df to the dict.
    # Then, check to see if the current is in the dataframe
    # already. If so, raise an error. If not, add a new column to the
    # dataframe corresponding to that current

    dict_of_dataframes = dict()
    input_column_list = input_dataframe.columns.tolist()

    for i, column_name in enumerate(input_column_list):
        if column_name != 'NoName':
            split_name = column_name.split(',')
            label = split_name[0]
            current = split_name[1]

            if label not in dict_of_dataframes:
                dict_of_dataframes[label] = pd.DataFrame()

            dataframe_to_modify = dict_of_dataframes[label]

            if current in dataframe_to_modify:
                raise ValueError("Excel sheet has multiple entries with the same label and current")
            else:
                column_to_add = input_dataframe.iloc[1:, i+1]
                dataframe_to_modify[current] = column_to_add

    return (time_axis, dict_of_dataframes)

def format_power_step(input_dataframe):
    """
    Take in the excel sheet for power step, and return a dictionary,
    where the key is the label, and the value is a dictionary containing
    currents associated with power levels.
    """

    # Loop over the first column (label), and add each to a dict if not
    # existant. Then, extract the current and add it to the sub-dict
    power_step_dict = dict()
    
    for i, label in enumerate(input_dataframe.iloc[:, 0]):
        if label not in power_step_dict:
            power_step_dict[label] = dict()

        labels_dict = power_step_dict[label]
        current = f'{input_dataframe.iloc[i, 1]}A'
        if current in labels_dict:
            raise ValueError("Excel sheet has multiple entries with the same label and current")
        else:
            power = input_dataframe.iloc[i, 2]
            labels_dict[current] = power

    return power_step_dict


"""
excel_file_path = "C:\\Users\\natha\\Alabama\\baker Research\\Void Study\\Experimental Data\\Void Study FULL DOC.xlsx"
dfs = pd.read_excel(excel_file_path, sheet_name=None)
zth_df = dfs["Zth"]
zth_formatted = rename_columns(zth_df, "NAHANS VOID STUDY")
dict_of_df = format_timed_data(zth_formatted)
pwr_df = dfs["Power Step"]
pwr_dict = format_power_step(pwr_df)
"""

