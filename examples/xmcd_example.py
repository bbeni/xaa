from xaa.core import SingleMeasurementProcessor
from xaa.operations import Normalize, CheckPoint, LineBGRemove,\
                           FermiBG, Integrate, SplitBy, Average,\
                           CombineDifference, Cut, CombineAverage,\
                           BackToNamed
from xaa.plotting import CheckpointPlotter
from xaa.preprocess import read_boreas_file_simple

import numpy as np


# which data columns should be extracted to dataframe
LABELS = [
    'energy_mono_encoder',
    'energy_mono_ct',
    'adc2_i3',
    'adc2_i2',
    'adc2_i1',
    'adc2_i4'
]

# add new columns with header parameter values
HEADER_PARAMS = [
    'magnet_z',
    'Mares_Cryo_temp',
    'ideu71_motor_polarization', # the X-Ray polarization
]

# rename columns
RENAMINGS = {
    'energy_mono_encoder':  'energy',
    'energy_mono_ct':       'energy', # if the first is not present use the second as 'energy'
    'Mares_Cryo_temp':      'cryo_temp',
    'ideu71_motor_polarization': 'polarization',
}

# normalize by reference signal
CALCULATIONS = {
    'tey': [np.divide, 'adc2_i3', 'adc2_i2'],
    'tfy': [np.divide, 'adc2_i4', 'adc2_i2']
}


# TODO: include stripped version of SH1_Tue09.dat
dataframes, numbers = read_boreas_file_simple(
    filename="./example_data/SH1_Tue09.dat",
    data_columns=LABELS,
    header_params=HEADER_PARAMS,
    renamings=RENAMINGS,
    calculations=CALCULATIONS
)

# now take measurement 263 to 271 for 30 NNMO//STO Mn L2,3 edge XMCD.
# @Temporary we have to subtract 2 because we skipped 1 measurement and start at 1
# TODO: ...
start, end = 261, 269
measurements = dataframes[start:end]

def is_left_polarized(df):
    return df.polarization[0] < 0

pipeline = [Cut,
            SplitBy(binary_filter=is_left_polarized),   # split data by left and right polarization
            Average,                                    # take average of left and right individually
            CheckPoint('raw'),                          #      Checkpoint named 'raw'
            LineBGRemove,                               # remove a fitted line fitted to 'line_range'
            FermiBG,
            CheckPoint('bg_removed'),                   #      Checkpoint named 'bg_removed'
            CombineAverage,                             # take average of left and right
            Normalize(save='mu0'),                      # normalize to maximum and save max as 'mu0'
            CheckPoint('normalized_XAS'),               #      Checkpoint
            Integrate,                                  # take integral numerically
            CheckPoint('integral_XAS'),                 #      Checkpoint
            BackToNamed('bg_removed'),                  # go to Checkpoint 'bg_removed' -> split data
            Normalize(to='mu0'),                        # normalize split data (divide both by 'mu0')
            CombineDifference,                          # left - right
            CheckPoint('XMCD'),                         #      Checkpoint named 'XMCD'
            Integrate,                                  # Integrate XMCD
            CheckPoint('integral_XMCD')]                #      Checkpoint named 'integral_XMCD'

# the parameters that the pipline needs
pipeline_parameters = {
    'cut_range':  (633, 667),
    'line_range': (627, 635),
    'peak_1':     (642, 647),
    'peak_2':     (652, 656),
    'post':       (660, 662),
    'a':          0.05,
    'delta':      1.5
}

p = SingleMeasurementProcessor(pipeline, pipeline_parameters)
p.add_data(measurements, x_column='energy', y_column='tey')
p.run()

# plots named checkpoints
cp_plotter = CheckpointPlotter(['normalized_XAS', 'XMCD', 'integral_XMCD'])
cp_plotter.plot(p)
