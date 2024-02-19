'''
Example from README

'''
import pandas as pd
from xaa.core import SingleMeasurementProcessor
from xaa.plotting import CheckpointPlotter
from xaa.operations import Normalize, CheckPoint, Integrate, \
    SplitBy, Average, CombineDifference, CombineAverage, BackToNamed

from xaa.operations import LineBGRemove

dataframes = [pd.read_csv('csvs/m{}.csv'.format(i)) for i in range(1,9)]

def is_left_polarized(df):
    return df.polarization[0] > 0

pipeline_params = {'line_range': [630, 635]} # we need to specify this because of the LineBgRemove Operation.

pipeline = [SplitBy(binary_filter=is_left_polarized),   # split data by left and right polarization
            Average,                                    # take average of left and right individually
            CheckPoint('raw'),                          #      Checkpoint named 'raw'
            LineBGRemove,                               # remove a fitted line fitted to 'line_range'
            CheckPoint('line_removed'),                 #      Checkpoint named 'line_removed'
            CombineAverage,                             # take average of left and right
            Normalize(save='mu0'),                      # normalize to maximum and save max as 'mu0'
            CheckPoint('normalized_xas'),               #      Checkpoint
            Integrate,                                  # take integral numerically
            CheckPoint('integral_xas'),                 #      Checkpoint
            BackToNamed('line_removed'),                # go to Checkpoint 'line_removed' -> split data
            Normalize(to='mu0'),                        # normalize split data (divide both by 'mu0')
            CombineDifference,                          # left - right
            CheckPoint('xmcd'),                         #      Checkpoint named 'xmcd'
            Integrate,                                  # Integrate xmcd
            CheckPoint('integral_xmcd')]                #      Checkpoint named 'integral_xmcd'

p = SingleMeasurementProcessor()
p.add_pipeline(pipeline)
p.add_params(pipeline_params)
p.check_missing_params()
p.add_data(dataframes, x_column='energy', y_column='XAS')
p.run()

# now the operations are applied to the data and we can retrieve any checkpoint with
print(p.get_checkpoint('xmcd'))

# we can also save everything to a csv
df = p.df_from_named()
df.to_csv('test_run.csv')

# or plot the results to see what happend for debugging
# give it a list of checkpoint names to plot.
plotter = CheckpointPlotter(['normalized_xas', 'xmcd', 'integral_xmcd', 'line_removed'])
plotter.plot(p)