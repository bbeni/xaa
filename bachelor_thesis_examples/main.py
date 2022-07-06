#!/usr/bin/env python3
import pandas as pd
import numpy as np

from xaa.core import SingleMeasurementProcessor, MultiMeasurementProcessor
from xaa.plotting import CheckpointPlotter, CheckpointPlotterMulti, TFYTEYComparePlot, OneAxisPlot
from xaa.operations import Normalize, CheckPoint, LineBG, FermiBG, BackTo, Integrate, SplitBy, Average, CombineDifference, Cut, CombineAverage, Add
from xaa.loaders.util import get_measurements_boreas_file

def test():
    # LAO Ni XLD
    dataframes = get_measurements_boreas_file('data_files/SH1_Tue09.dat', range(42+1,  46+1))

    pipeline = [LineBG, Average, FermiBG, Normalize(save='m1'),
                BackTo(0), SplitBy, Average, Normalize(to='m1'), LineBG, CheckPoint, FermiBG, CheckPoint,
                CombineDifference, CheckPoint, Integrate]

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
                CombineDifference, CheckPoint, Integrate, CheckPoint]

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
                CombineDifference, CheckPoint, Integrate, CheckPoint]

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
                CombineDifference, CheckPoint, Integrate, CheckPoint]

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

    def filter_circular(df):
        return df.polarization[0] < 0

    pipeline = [Cut, Average, Normalize(save='m1'), BackTo(1), SplitBy, Normalize(to='m1'), Average, CombineAverage, CheckPoint]
    global_params = {'cut_range':cut_range,
                     'binary_filter':filter_circular,
                     }

    p = SingleMeasurementProcessor(); p.add_pipeline(pipeline); p.add_params(global_params); p.check_missing_params()
    p.add_data(dataframes, x_column='energy', y_column='mu_normalized')
    p.run()
    smpTEY = p

    pipeline.insert(-2, Add(-0.5)); pipeline.insert(-2, Normalize)
    p = SingleMeasurementProcessor(); p.add_pipeline(pipeline); p.add_params(global_params); p.check_missing_params()
    p.add_data(dataframes, x_column='energy', y_column='tfy_normalized')
    p.run()
    smpTFY = p

    plotter = TFYTEYComparePlot()
    x1, y1 = smpTEY.get_checkpoint(-1)
    x2, y2 = smpTFY.get_checkpoint(-1)
    plotter.plot_xy(x1, x2, y1, y2)


def fermi_bg_example():
    #data_range = range(263+1, 271+1) # mn // sto xmcd low temp
    data_range = range(335+1, 343+1) # ni // sto xmcd low temp

    def filter_circular(df):
        return df.polarization[0] < 0

    pipeline_params = {'binary_filter': filter_circular,
                       'line_range': (835, 845),
                       'peak_1':(853, 854.2),
                       'peak_2':(871.4, 872.5),
                       'post':(877, 884),
                       'delta':0.3,
                       'a':2/3
                       }

    dataframes = get_measurements_boreas_file('data_files/SH1_Tue09.dat', data_range)


    smp = SingleMeasurementProcessor()
    pipeline = [Average, Normalize(save='m1'), BackTo(0), SplitBy, Normalize(to='m1'),
               Average, CombineAverage, CheckPoint, LineBG, CheckPoint, FermiBG, CheckPoint]

    #pipeline = [Average, CheckPoint]

    smp.add_pipeline(pipeline)
    smp.add_params(pipeline_params)
    smp.check_missing_params()
    smp.add_data(dataframes, x_column='energy', y_column='mu_normalized')
    smp.run()

    plot = OneAxisPlot(n_colors=4, xlabel='Photon Energy [eV]', ylabel='normalized $\mu$ (arb.u.)',
                       figsize=(3.54, 3), axis_dimensions=(0.18, 0.15, 0.79, 0.82))

    x, y = smp.get_checkpoint(0)
    #plt.plot(x, y, label='before line bg is removed(raw)')
    x1, y1 = smp.get_checkpoint(1)
    plot.plot(x1, y1, label='raw signal normalized')
    x2, y2 = smp.get_checkpoint(2)
    plot.plot(x2, y1-y2, label='fermi step $\\delta$ = 0.3', color_nr=0)

    for i, delta in enumerate([1.0, 2.0, 3.0]):
        pipeline_params['delta']=delta
        smp = SingleMeasurementProcessor()
        smp.add_pipeline(pipeline)
        smp.add_params(pipeline_params)
        smp.check_missing_params()
        smp.add_data(dataframes, x_column='energy', y_column='mu_normalized')
        smp.run()
        x, y1 = smp.get_checkpoint(1)
        _, y2 = smp.get_checkpoint(2)
        plot.plot(x, y1-y2, label='fermi step $\\delta$ = {}'.format(delta), color_nr=i+1)


    plot.xlim((845, 885))
    plot.ylim((-0.05, 0.1))
    plot.finish(save='../bachelor-thesis/xaa_figures/fermi_step.pdf')



if __name__ == "__main__":
    #test()
    #test2()
    #strain_ni_xld_test()
    #thickness_ni_xld_fy_test()
    #tfy_vs_tey()
    fermi_bg_example()
