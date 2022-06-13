#!/usr/bin/env python3
import pandas as pd
import numpy as np

from core import SingleMeasurementProcessor, MultiMeasurementProcessor
from plotting import CheckpointPlotter, CheckpointPlotterMulti
from operations import Normalize, CheckPoint, LineBG, FermiBG, BackTo, Integrate, SplitBy, Average, Difference, Cut, CombineAverage
from loaders.util import get_measurements_boreas_file

def test():
    # LAO Ni XLD
    dataframes = get_measurements_boreas_file('data_files/SH1_Tue09.dat', range(42+1,  46+1))

    pipeline = [LineBG, Average, FermiBG, Normalize(save='m1'),
                BackTo(0), SplitBy, Average, Normalize(to='m1'), LineBG, CheckPoint, FermiBG, CheckPoint,
                Difference, CheckPoint, Integrate]

    p = SingleMeasurementProcessor()
    p.add_pipeline(pipeline)

    filter_horizontal = lambda df: df.polarization[0] < np.pi/4

    global_params = {'line_range': [627, 635],
                     'binary_filter':filter_horizontal,
                     'peak_1': (642, 647),
                     'peak_2': (652, 656),
                     'post': (660, 661),
                     'a': 0, 'delta': 1.5}

    p.add_params(global_params)
    p.check_missing_params()

    p.add_data(dataframes, x_column='energy', y_column='tfy_normalized')
    p.run()

    plotter = CheckpointPlotter()
    plotter.plot(p)

def test2():
    dataframes = get_measurements_boreas_file('data_files/SH1_Tue09.dat', [range(42+1,  46+1), range(47+1, 51+1)])

    pipeline = [LineBG, CheckPoint, Average, FermiBG, CheckPoint, Normalize(save='m1'),
                BackTo(0), SplitBy, Average, Normalize(to='m1'), LineBG, FermiBG, CheckPoint,
                Difference, CheckPoint, Integrate, CheckPoint]

    p = MultiMeasurementProcessor(2)
    p.add_pipeline(pipeline)

    filter_horizontal = lambda df: df.polarization[0] < np.pi/4

    global_params = {'line_range': [627, 635],
                     'binary_filter':filter_horizontal,
                     'peak_1': (642, 647),
                     'peak_2': (652, 656),
                     'post': (660, 661),
                     'a': 0, 'delta': 1.5}

    p.add_params(global_params)
    p.check_missing_params()

    p.add_data(dataframes, x_column='energy', y_column='mu_normalized')
    p.run()

    print()
    plotter = CheckpointPlotterMulti()
    plotter.plot(p)

def strain_ni_xld_test():
    '''300 Kelvin 0 Tesla'''
    labels = ['LAO', 'NGO', 'LSAT', 'LGO', 'STO']
    measurement_indices = [(87,   91), (92,   96), (97,  101), (107, 111), (102, 106)]
    indices_ranges = [range(a+1, b+1) for a, b in measurement_indices]

    ni_params = {'line_range': (840, 848),
                     'peak_1': (853, 855),
                     'peak_2': (866, 875),
                     'pre': (840, 848),
                     'post': (878, 883),
                     'a': 1/3, 'delta': 1.5}

    dataframes = get_measurements_boreas_file('data_files/SH1_Tue09.dat', indices_ranges)

    filter_horizontal = lambda df: df.polarization[0] < np.pi/4
    pipeline = [Average, LineBG, FermiBG, Normalize(save='m1'),
                BackTo(0), SplitBy(filter_horizontal), Average, Normalize(to='m1'), LineBG, FermiBG, CheckPoint,
                Difference, CheckPoint, Integrate, CheckPoint]

    #p = MultiMeasurementProcessor(5)
    p = SingleMeasurementProcessor()
    p.add_pipeline(pipeline)
    p.add_params(ni_params)
    p.check_missing_params()

    p.add_data(dataframes[0], x_column='energy', y_column='tfy_normalized')
    p.run()

    plotter = CheckpointPlotter()
    plotter.plot(p)

def thickness_ni_xld_fy_test():

    thickness = [3, 6, 10, 21, 30]  # unit cells
    measurement_indices = [(102, 106), (122, 126), (117, 121), (127, 131), (112, 116),]
    indices_ranges = [range(a+1, b+1) for a, b in measurement_indices]


    ni_params = {'line_range': (840, 848),
                     'peak_1': (853, 855),
                     'peak_2': (866, 875),
                     'pre': (840, 848),
                     'post': (878, 883),
                     'a': 1/3, 'delta': 1.5}

    dataframes = get_measurements_boreas_file('data_files/SH1_Tue09.dat', indices_ranges)

    filter_horizontal = lambda df: df.polarization[0] < np.pi/4
    pipeline = [Average, LineBG, FermiBG, Normalize(save='m1'),
                BackTo(0), SplitBy(filter_horizontal), Average, Normalize(to='m1'), LineBG, FermiBG, CheckPoint,
                Difference, CheckPoint, Integrate, CheckPoint]

    #p = MultiMeasurementProcessor(5)
    p = SingleMeasurementProcessor()
    p.add_pipeline(pipeline)
    p.add_params(ni_params)
    p.check_missing_params()

    p.add_data(dataframes[0], x_column='energy', y_column='mu_normalized')
    p.run()

    plotter = CheckpointPlotter()
    plotter.plot(p)

def xas_low_temp():
    pass

def tfy_vs_tey():

    #data_range = range(57+1,  60+1)
    #data_range = range(407+1, 415+1) # nd // sto xmcd low temp
    data_range = range(263+1, 271+1) # mn // sto xmcd low temp
    cut_range = [634, 660]

    dataframes = get_measurements_boreas_file('data_files/SH1_Tue09.dat', data_range)

    filter_circular = lambda df: df.polarization[0] < 0

    pipeline = [Cut, Average, Normalize(save='m1'), BackTo(1), SplitBy, Normalize(to='m1'), Average, CombineAverage, CheckPoint]
    global_params = {'cut_range':cut_range,
                     'binary_filter':filter_circular,
                     }

    for yiel in ['tfy_normalized', 'mu_normalized']:
        p = SingleMeasurementProcessor()
        p.add_pipeline(pipeline)

        p.add_params(global_params)
        p.check_missing_params()

        p.add_data(dataframes, x_column='energy', y_column=yiel)
        p.run()

        plotter = CheckpointPlotter()
        plotter.plot(p)


if __name__ == "__main__":
    #test()
    #test2()
    #strain_ni_xld_test()
    #thickness_ni_xld_fy_test()
    tfy_vs_tey()
