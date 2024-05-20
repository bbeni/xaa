''' This module provides a function for reading the .dat file produced
    at the boreas beamline in alba synchrotron, see website:
        https://www.cells.es/en/beamlines/bl29-boreas

    It is probably applicable to other beamlines data (but I didn't check that yet).

    The file consists of text structured as follows:
        For now see the example in examples/example_data
        TODO: specify file format
'''
import re
import pandas as pd
import numpy as np

def read_boreas_file_simple(filename, data_columns, header_params=[], renamings={}, calculations={}):
    """ Read the .dat file and 
        return a list of pandas dataframes with one acquisition each dataframe 
    
       filename:      .dat file from beamline (see examples)
       data_columns:  the rows of measured data that we want as a list
       header_params: the parameters in the header of a measurement you want as a row 
                      (will be constant and set for all data points - it's kinda stupid but convenient)
       renamings:     dictionary of labels from data_rows or header_params to rename.
       calculations:  example:
                      calculations = {
                          "tey_normalized": [np.divide, 'adc2_i3', 'adc2_i2'],
                      }
    """

    if type(header_params) != list:
        print("ERROR: header_params must be a list of strings!")
        return None, None

    acquisition_regex = r'(#S.*?)#L\s+(.*?)(?:\Z|#C)'
    number_date_regex = r'#S\s+(.*?)\s+.*?#D\s+(.*?)\n#C'

    acquisition_regex = re.compile(acquisition_regex, re.M ^ re.DOTALL)
    number_date_regex = re.compile(number_date_regex, re.M ^ re.DOTALL)


    with open(filename, "r") as f:
        content = f.read()

    acquisitions = re.findall(acquisition_regex, content)

    dfs = []
    ac_numbers = []
    dates = []

    for header, data_block in acquisitions:

        # find acquisition number and date
        m = re.match(number_date_regex, header)
        if m:
            ac_number, date = m.groups()
            ac_numbers.append(ac_number)
            dates.append(date)
        else:
            print("ERROR: have no title! didn't match anything. skipped.")
            continue


        # read the data block
        lines = data_block.splitlines()
        column_names = lines[0].split()
        if lines[1:] == []:
            print(f"ERROR: couldn't read data of acquisition nr ({ac_number}). skipped.")
            ac_numbers = ac_numbers[:-1]
            continue
        else:
            df = pd.DataFrame(np.genfromtxt(lines[1:], names=column_names))
        

        # filter the columns
        valid_cols = [col for col in data_columns if col in df]
        if len(valid_cols) == 0:
            print("WARNING: we have found 0 of the columns you provided in data_columns! Taking all the data cols.")
        else:
            df = df[valid_cols]


        # inject labeled header_parameters into the dataframe as a column
        if len(header_params) != 0:

            # find parameters
            param_strs = re.findall(r'#O\d+\s+(.*?)\n', header)
            p_labels = [s for line in param_strs for s in line.split()]

            # find parameter values
            param_values = re.findall(r'#P\d+\s+(.*?)\n', header)
            p_values = [float(s) for line in param_values for s in line.split()]

            for p_label in header_params:
                try:
                    p_labels.index(p_label)
                except ValueError:
                    print(f"ERROR: didn't find the header label ({p_label}). skipped.")
                    continue

                df[p_label] = p_values[p_labels.index(p_label)]

        # update df from user defined CALCULATIONS
        for c_label, c in calculations.items():
            f = c[0]
            args = c[1:]
            args = map(lambda label: df[label], args)
            df[c_label] = f(*args)

        # update df from user defined RENAMINGS
        if renamings != {}:
            for name, new_name in renamings.items():
                df = df.rename({name:new_name}, axis='columns')

        dfs.append(df)

    return dfs, ac_numbers

if __name__ == "__main__":
    # @Temporary Example
    # TODO: implement commandline parsing
    dfs, _ = read_boreas_file_simple(
        filename = "./example_data/SH1_Tue09.dat",
        data_columns = ['energy_mono_encoder', 'energy_mono_ct', 'adc2_i3', 'adc2_i2', 'adc2_i1', 'adc2_i4'],
        header_params = ['magnet_z', 'Mares_Cryo_temp', 'ideu71_motor_polarization'],
        renamings = {
            'energy_mono_encoder': 'energy',
            'energy_mono_ct': 'energy',
            'Mares_Cryo_temp': 'cryo_temp',
            'ideu71_motor_polarization': 'polarization'
        },
        calculations = {
            'mu_normalized':  [np.divide, 'adc2_i3', 'adc2_i2'],
            'tfy_normalized': [np.divide, 'adc2_i4', 'adc2_i2']
        }
    )

    print("Read", len(dfs), "acquisitions.")


'''    
##########################################
# PREPROCESS CONFIGURATION FOR MY THESIS #
##########################################

# labels to extract (which ones are interesting/needed?) -> only those get saved into the dataframes
# leave empty to extract all
LABELS = ['energy_mono_encoder', 'energy_mono_ct', 'adc2_i3', 'adc2_i2', 'adc2_i1', 'adc2_i4']

# rename columns
# leave empty to not rename anything
RENAMINGS = {
    'energy_mono_encoder' : 'energy',
    'energy_mono_ct':       'energy'
}

# save parameters from header comment of a measurement
# it will create a column in the dataframe with the selected parameter
# the header is of the form '#P1 . . x .'
# to get the x value and rename it like this: 'new_name': 'x'
PARAMS = {
    'magnet_z':     'magnet_z',
    'cryo_temp':    'Mares_Cryo_temp',
    'polarization': 'ideu71_motor_polarization',
}

# define some preprocessing calculations done on the dataframe
# every entry creates a new column
CALCULATIONS = {
    'mu_normalized':  [np.divide, 'adc2_i3', 'adc2_i2'],
    'tfy_normalized': [np.divide, 'adc2_i4', 'adc2_i2']
}
'''