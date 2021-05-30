#!/usr/bin/env python3
import pandas as pd
import numpy as np

from core import SingleMeasurementProcessor
from plotting import CheckpointPlotter
from operations import Normalize, CheckPoint, LineBG, FermiBG, BackTo, Integrate, SplitBy, Average, Difference

from loaders.util import get_measurements_boreas_file



# another requirement is that we have estimated parameters in case nothing is provided

# we can structure it like a tree
# pipline examples:
#  single dfs -> select_x,y -> average -> normalize(save='m1')
#                           -> split_by(filter) -> average -> normalize(to='m1') -> difference -> integrate -> integral_gather
#  single dfs -> slect_x,y -> average -> bg_subtract(method) -> normalize -> vary('x', [1,2]) -> bg_subtract('fermi') -> integrate -> vary -> integral_gather

def test():

    dataframes = get_measurements_boreas_file('data_files/SH1_Tue09.dat', range(42,  46+1))


    pipeline = [LineBG, FermiBG(a=0, delta=1.5), Average, Normalize(save='m1'), BackTo(2), Normalize(to='m1'), SplitBy, Average, Difference, CheckPoint, Integrate, CheckPoint]
    p = SingleMeasurementProcessor()
    p.add_pipeline(pipeline)


    filter_horizontal = lambda df: df.polarization[0] < np.pi/4

    global_params = {'line_range': [627, 635],
                     'binary_filter':filter_horizontal,
                     'peak_1': (642, 647),
                     'peak_2': (652, 656),
                     'post': (660, 661),}

    p.add_params(global_params)
    p.missing_params()


    p.add_data(dataframes, x_column='energy', y_column='mu_normalized')
    p.run()

    plotter = CheckpointPlotter()
    plotter.plot(p)

    #pipeline1 = [SelectXY, Interpolate, Average, Normalize(save='m1'), Back(steps=2), SplitBy(f=left), Average, Normalize(to='m1'), Difference, Integrate]

if __name__ == "__main__":
    test()
