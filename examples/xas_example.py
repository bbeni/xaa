from xaa.core import       SingleMeasurementProcessor
from xaa.operations import Normalize, CheckPoint, LineBGRemove, FermiBG, BackTo, Integrate,\
                           Average, Cut, CombineAverage, Add, Flip, ApplyFunctionToY, BackToNamed
from xaa.plotting import   CheckpointPlotter
from xaa.preprocess import read_boreas_file_simple

import numpy as np
import pandas as pd

# TODO: include stripped version of ./SH1_Tue09.dat

def extract_measurements_dat_file():

    dataframes, numbers = read_boreas_file_simple(
        filename=     './example_data/SH1_Tue09.dat',
        data_columns= ['energy_mono_encoder', 'energy_mono_ct', 'adc2_i3', 'adc2_i2', 'adc2_i1', 'adc2_i4'],
        header_params=['magnet_z', 'Mares_Cryo_temp', 'ideu71_motor_polarization'],
        renamings={
            'energy_mono_encoder' : 'energy',
            'energy_mono_ct':       'energy',
            'cryo_temp':            'Mares_Cryo_temp',
            'polarization':         'ideu71_motor_polarization'
        },
        calculations={
            'tey': [np.divide, 'adc2_i3', 'adc2_i2'],
            'tfy': [np.divide, 'adc2_i4', 'adc2_i2'],
        })

    for df, i in zip(dataframes, numbers):
        df.to_csv(f"csvs/m{i}.csv")

    return dataframes, numbers
#
# only run this once, because it takes some time.
# afterwards you can comment it out!
#
dataframes, _ = extract_measurements_dat_file()

# 
# now load the file and measurement 263 to 271
# this is sample 30 uc Nd2NiMnO6 // SrTiO3
#
measurements = list(range(263, 271))
dataframes = [pd.read_csv(f"csvs/m{i}.csv") for i in measurements]

pipeline = [CheckPoint("raw"),
            Average,
            CheckPoint("raw_avg"),
            LineBGRemove,
            FermiBG,
            Normalize,
            CheckPoint('normalized_xas'),
            Integrate,
            CheckPoint('integral_xas')]

pipeline_params = {'line_range': (627, 635),
                   'peak_1':     (642, 647),
                   'peak_2':     (652, 656),
                   'post':       (660, 662),
                   'a':          0.05,
                   'delta':      1.5}

p = SingleMeasurementProcessor(pipeline, pipeline_params)
p.add_data(dataframes, x_column='energy', y_column='tey')
p.run()

cp_plotter = CheckpointPlotter(["raw", "raw_avg"])
cp_plotter.plot(p)

cp_plotter = CheckpointPlotter(["normalized_xas", "integral_xas"])
cp_plotter.plot(p)
