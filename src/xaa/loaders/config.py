"""Author: Beni

configure which preprocessing is done on the .dat file
"""
import numpy as np

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
