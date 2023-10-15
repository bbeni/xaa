import xaa.loaders.config
from xaa.core import SingleMeasurementProcessor
from xaa.operations import Normalize, CheckPoint, LineBGRemove, FermiBG, BackTo, Integrate,\
    SplitBy, Average, CombineDifference, Cut, CombineAverage, Add, Flip, ApplyFunctionToY, BackToNamed
from xaa.plotting import CheckpointPlotter

import numpy as np


# which data should be extracted to dataframe
xaa.loaders.config.LABELS = [
    'energy_mono_encoder',
    'energy_mono_ct',
    'adc2_i3',
    'adc2_i2',
    'adc2_i1',
    'adc2_i4']

# rename columns
xaa.loaders.config.RENAMINGS = {
    'energy_mono_encoder' : 'energy',
    'energy_mono_ct': 'energy'
}

# add new columns with header parameter values
xaa.loaders.config.PARAMS = {
    'magnet_z': 'magnet_z',
    'cryo_temp': 'Mares_Cryo_temp',
    'polarization': 'ideu71_motor_polarization',
}

# do normalization
xaa.loaders.config.CALCULATIONS = {
    'mu_normalized': [np.divide, 'adc2_i3', 'adc2_i2'],
    'tfy_normalized': [np.divide, 'adc2_i4', 'adc2_i2']
}

# now load the file and measurement 263 to 271
measurements = list(range(263, 271+1))

from xaa.loaders.util import boreas_file_to_dataframes
dataframes = boreas_file_to_dataframes('./SH1_Tue09.dat', measurements)





def filter_circular(df):
    return df.polarization[0] < 0

# now setup the pipline to calculate the xmcd
# xmcd pipelines Checkpoint 1 is the integral mu+ + mu-, Checkpoint 3 is the integral mu+ - mu-
pipeline = [SplitBy(binary_filter=filter_circular), Average, LineBGRemove, FermiBG, CheckPoint('left_right'),
            CombineAverage, Normalize(save='mu0'), CheckPoint('normalized_xas'),
            Integrate, CheckPoint('integral_xas'),
            BackToNamed('left_right'), Normalize(to='mu0'), CombineDifference, CheckPoint('xmcd'),
            Integrate, CheckPoint('integral_xmcd')]

# the parameters that the pipline needs in order to fit stuff
pipeline_params = {'cut_range': (627, 662),
                   'line_range': (627, 635),
                   'peak_1': (642, 647),
                   'peak_2': (652, 656),
                   'post': (660, 662),
                   'a': 0.05, 'delta': 1.5}


p = SingleMeasurementProcessor()
p.add_pipeline(pipeline)
p.add_params(pipeline_params)
p.check_missing_params()
p.add_data(dataframes, x_column='energy', y_column='mu_normalized')
p.run()


# plots every named checkpoint
cp_plotter = CheckpointPlotter()
cp_plotter.plot(p)
