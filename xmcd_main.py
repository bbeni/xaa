#!/usr/bin/env python3
import pandas as pd
import numpy as np

from core import SingleMeasurementProcessor, MultiMeasurementProcessor
from plotting import CheckpointPlotter, CheckpointPlotterMulti, TFYTEYComparePlot, OneAxisPlot
from operations import Normalize, CheckPoint, LineBG, FermiBG, BackTo, Integrate,\
    SplitBy, Average, CombineDifference, Cut, CombineAverage, Add, Flip, ApplyFunctionToY
from loaders.util import get_measurements_boreas_file

labels_name = ['LAO', 'NGO', 'LSAT', 'LGO', 'STO', '3uc', '6uc', '10uc', '21uc']
labels_id = ['f1', 'f2', 'f3', 'f5', 'f4', 'b2', 'b1', 'b3', 'f6']

table_xmcd_ni_high_b = [
    [[308,316], 'f1', 2, 5],
    [[317,325], 'f2', 2, 5],
    [[326,334], 'f3', 2, 5],
    [[344,352], 'f5', 2, 5],
    [[335,343], 'f4', 2, 5],
    [[371,379], 'b2', 2, 5],
    [[362,370], 'b1', 2, 5],
    [[224,232], 'b3', 2, 5],
    [[353,361], 'f6', 2, 5]]

table_xmcd_mn_high_b = [
    [[236,244], 'f1', 2, 5],
    [[245,253], 'f2', 2, 5],
    [[254,262], 'f3', 2, 5],
    [[272,280], 'f5', 2, 5],
    [[263,271], 'f4', 2, 5],
    [[299,307], 'b2', 2, 5],
    [[290,298], 'b1', 2, 5],
    [[206,214], 'b3', 2, 5],
    [[281,289], 'f6', 2, 5]]

cp_plotter = CheckpointPlotter()

filter_circular = lambda df: df.polarization[0] > 0

#xmcd pipelines Checkpoint 1 is the integral mu+ + mu-, Checkpoint 3 is the integral mu+ - mu-
pipeline_basic_TEY_xmcd = [SplitBy(binary_filter=filter_circular), Average, LineBG, FermiBG,
                           CombineAverage, Normalize(save='mu0'), CheckPoint,
                           Integrate, CheckPoint,
                           BackTo(to=4), Normalize(to='mu0'), CombineDifference, CheckPoint,
                           Integrate, CheckPoint]

pipeline_basic_TFY_xmcd = [SplitBy(filter_circular), Average, ApplyFunctionToY(function=lambda y: 1/y), LineBG, FermiBG,
                           CombineAverage, Normalize(save='mu0'), CheckPoint,
                           Integrate, CheckPoint,
                           BackTo(to=5), Normalize(to='mu0'), CombineDifference, CheckPoint,
                           Integrate, CheckPoint]


def process_measurement(measurements_range, pipeline, pipeline_params, y_column):
    dataframes = get_measurements_boreas_file('data_files/SH1_Tue09.dat', measurements_range)

    p = SingleMeasurementProcessor()
    p.add_pipeline(pipeline)

    p.add_params(pipeline_params)
    p.check_missing_params()
    p.add_data(dataframes, x_column='energy', y_column=y_column)
    p.run()
    return p


def thole_cara_lz(rho, n, l, c):
    return rho * 2 * (l * (l + 1) * (4 * l + 2 - n)) / (c * (c + 1) - l * (l + 1) - 2)


def thole_cara_sz(int1, int2, int_plus, n, l, c):
    # neglect Tz
    delta = (int1 - (c + 1) / c * int2) / int_plus
    return delta * 3 * c * (4 * l + 2 - n) / (l * (l + 1) - 2 - c * (c + 1))


def sum_rules_basic_pipline(processor, mid_index, n, l, c):
    # sum rules by thole and cara lz
    int_minus = processor.get_checkpoint(3)[1][-1]
    int_plus = 3 * processor.get_checkpoint(1)[1][-1]
    rho = int_minus / int_plus
    lz = thole_cara_lz(rho=int_minus / int_plus, n=n, l=l, c=c)

    # sum rules by thole and cara sz
    int1 = processor.get_checkpoint(3)[1][-1] - processor.get_checkpoint(3)[1][mid_index]
    int2 = processor.get_checkpoint(3)[1][mid_index]
    sz = thole_cara_sz(int1=int1, int2=int2, int_plus=int_plus, n=2, l=2, c=1)

    return lz, sz

def ni_xmcd_thickness():
    # ni xmcd thickness dependence

    ni_thick_params = {#'cut_range': [835, 877],
                       'line_range': [835, 845],
                       'binary_filter': filter_circular,
                       'peak_1': (853, 854.5),
                       'peak_2': (871.48, 871.51),
                       'post': (875, 884.5),
                       'a': 2 / 3, 'delta': 1.5}

    lz = []
    sz = []

    for (a, b), _, _, _ in table_xmcd_ni_high_b[-5:]:
        measurements_range = range(a+1, b+1)
        p = process_measurement(measurements_range, pipeline_basic_TEY_xmcd, ni_thick_params, 'mu_normalized')

        mid = int(3/5 *(len(p.get_checkpoint(3)[1])))
        Lz, Sz = sum_rules_basic_pipline(processor=p, mid_index=mid, n=2, l=2, c=1)
        sz.append(Sz)
        lz.append(Lz)

        cp_plotter.plot(p)

    print("Lz 30 uc, 3 uc, 6 uc, 10 uc, 21 uc", lz)
    print("Sz 30 uc, 3 uc, 6 uc, 10 uc, 21 uc", sz)

    x = [30, 3, 6, 10, 21]
    def fermi(x, c1, c2, c3, c4): return c1/(1+np.exp((x-c2)/c3))-c4
    import matplotlib.pyplot as plt
    plt.scatter(x, lz, label='lz')
    plt.scatter(x, sz, label='sz')
    x_space = np.linspace(0, 100, 100)
    plt.plot(x_space, fermi(x_space, 7.8, 0, -17, 3), 'g')
    plt.plot(x_space, fermi(x_space, 7.4, 0, -17, 3), 'g--')
    plt.plot(x_space, fermi(x_space, 8.2, 0, -17, 3), 'g--')
    plt.scatter(x, np.array(lz)+np.array(sz), label='total')
    plt.legend()
    plt.show()

def ni_xmcd_strain():
    # ni xmcd strain dependence

    ni_thick_params = {'line_range': [835, 845],
                       'binary_filter': filter_circular,
                       'peak_1': (853, 854.5),
                       'peak_2': (871.48, 871.51),
                       'post': (882, 884.5),
                       'a': 1 / 3, 'delta': 1.5}

    lz = []
    sz = []

    for (a, b), _, _, _ in table_xmcd_ni_high_b[:5]:
        measurements_range = range(a + 1, b + 1)
        p = process_measurement(measurements_range, pipeline_basic_TEY_xmcd, ni_thick_params, 'mu_normalized')

        mid = int(3 / 5 * (len(p.get_checkpoint(3)[1])))
        Lz, Sz = sum_rules_basic_pipline(processor=p, mid_index=mid, n=2, l=2, c=1)
        sz.append(Sz)
        lz.append(Lz)

        #cp_plotter.plot(p)

    print("Lz", 'LAO', 'NGO', 'LSAT', 'LGO', 'STO', lz)
    print("Sz", 'LAO', 'NGO', 'LSAT', 'LGO', 'STO', sz)


def mn_xmcd_thickness():

    mn_thick_params = {'line_range': (627, 635),
                       'binary_filter': filter_circular,
                       'peak_1': (642, 647),
                       'peak_2': (652, 656),
                       'post': (660, 661),
                       'a': 1, 'delta': 1.5 }

    lz = []
    sz = []

    for (a, b), _, _, _ in table_xmcd_mn_high_b[-5:]:
        measurements_range = range(a+1, b+1)
        p = process_measurement(measurements_range, pipeline_basic_TEY_xmcd, mn_thick_params, 'mu_normalized')

        cp_plotter.plot(p)

    print("Lz 30 uc, 3 uc, 6 uc, 10 uc, 21 uc", lz)
    print("Sz 30 uc, 3 uc, 6 uc, 10 uc, 21 uc", sz)

def main():
    ni_xmcd_thickness()
    #ni_xmcd_strain()
    #mn_xmcd_thickness()

if __name__ == '__main__':
    main()