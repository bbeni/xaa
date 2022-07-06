#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

from xaa.core import SingleMeasurementProcessor, MultiMeasurementProcessor
from xaa.plotting import CheckpointPlotter, CheckpointPlotterMulti, TFYTEYComparePlot, OneAxisPlot
from xaa.operations import Normalize, CheckPoint, LineBG, FermiBG, BackTo, Integrate,\
    SplitBy, Average, CombineDifference, Cut, CombineAverage, Add, Flip, ApplyFunctionToY, BackToNamed
from xaa.loaders.util import get_measurements_boreas_file
from xaa.helpers import closest_idx

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

filter_circular = lambda df: df.polarization[0] < 0

#xmcd pipelines Checkpoint 1 is the integral mu+ + mu-, Checkpoint 3 is the integral mu+ - mu-
pipeline_basic_TEY_xmcd = [SplitBy(binary_filter=filter_circular), Average, LineBG, FermiBG, CheckPoint('left_right'),
                           CombineAverage, Normalize(save='mu0'), CheckPoint('normalized_xas'),
                           Integrate, CheckPoint('integral_xas'),
                           BackToNamed('left_right'), Normalize(to='mu0'), CombineDifference, CheckPoint('xmcd'),
                           Integrate, CheckPoint('integral_xmcd')]

pipeline_basic_TFY_xmcd = [SplitBy(filter_circular), Average, ApplyFunctionToY(function=lambda y: 1/y), LineBG, FermiBG, Flip, CheckPoint('left_right'),
                           CombineAverage, Normalize(save='mu0'), CheckPoint('normalized_xas'),
                           Integrate, CheckPoint('integral_xas'),
                           BackToNamed('left_right'), Normalize(to='mu0'), CombineDifference, CheckPoint('xmcd'),
                           Integrate, CheckPoint('integral_xmcd')]

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


def guo_gupta_lz(int_xmcd, int_plus, n_3d):
    return -4/3*int_xmcd[-1]/int_plus * (10-n_3d)


def guo_gupta_sz(int_xmcd, int_plus, mid, n_3d):
    intL3 = int_xmcd[mid] - int_xmcd[0]
    intL2 = int_xmcd[-1] - int_xmcd[mid]
    return -1 * (2*intL3 - 4*intL2)/int_plus * (10 - n_3d)


def sum_rules_basic_pipline(processor, mid_index, n, l, c, n_3d):
    int_xmcd = processor.get_named_checkpoint('integral_xmcd')[1]
    int_xas = processor.get_named_checkpoint('integral_xas')[1]

    # sum rules by thole and cara lz
    int_minus = int_xmcd[-1]
    int_plus = 3 * int_xas[-1]
    lz = thole_cara_lz(rho=int_minus / int_plus, n=n, l=l, c=c)

    # sum rules by thole and cara sz
    int1 = int_xmcd[-1] - int_xmcd[mid_index]
    int2 = int_xmcd[mid_index]
    sz = thole_cara_sz(int1=int1, int2=int2, int_plus=int_plus, n=2, l=2, c=1)

    # sum rule guo gupta
    sz_g = guo_gupta_sz(int_xmcd, int_plus, mid_index, n_3d)
    lz_g = guo_gupta_lz(int_xmcd, int_plus, n_3d)

    return lz, sz, lz_g, sz_g


def print_sumrules_thick(lz, sz, lzg, szg):
    print('\nthole cara sum rules')
    print("Lz 30 uc, 3 uc, 6 uc, 10 uc, 21 uc", lz)
    print("Sz 30 uc, 3 uc, 6 uc, 10 uc, 21 uc", sz)
    print('\nguo gupta reference')
    print("Lz 30 uc, 3 uc, 6 uc, 10 uc, 21 uc", lzg)
    print("Sz 30 uc, 3 uc, 6 uc, 10 uc, 21 uc", szg)


def print_sumrules_strain(lz, sz, lzg, szg):
    print('\nthole cara sum rules')
    print("Lz 'LAO', 'NGO', 'LSAT', 'LGO', 'STO'", lz)
    print("Sz 'LAO', 'NGO', 'LSAT', 'LGO', 'STO'", sz)
    print('\nguo gupta reference')
    print("Lz 'LAO', 'NGO', 'LSAT', 'LGO', 'STO'", lzg)
    print("Sz 'LAO', 'NGO', 'LSAT', 'LGO', 'STO'", szg)


index_strain = ['LAO', 'NGO', 'LSAT', 'LGO', 'STO']
index_thick = [30, 3, 6, 10, 21] # uc


def save_sumrules_csv(lz, sz, lzg, szg, fname, index):
    data = np.array([lz, sz, lzg, szg]).T
    df = pd.DataFrame(data, columns=['Lz', 'Sz', 'Lz_guo', 'Sz_guo'], index=index)
    df.to_csv('out/compare_data' + os.sep + fname + '.csv')


def ni_xmcd_thickness():
    # ni xmcd thickness dependence

    ni_thick_params = {#'cut_range': [835, 877],
                       'line_range': [835, 845],
                       'binary_filter': filter_circular,
                       'peak_1': (853, 854.5),
                       'peak_2': (871.48, 871.51),
                       'post': (875, 884.5),
                       'a': 2 / 3, 'delta': 1.5}

    n = 2
    l = 2
    c = 1
    n_3d = 8.2 # cluster model analysis reference 27 guo gupta

    lz = []
    sz = []
    lzg = []
    szg =[]

    for (a, b), label, _, _ in table_xmcd_ni_high_b[-5:]:
        measurements_range = range(a+1, b+1)
        p = process_measurement(measurements_range, pipeline_basic_TEY_xmcd, ni_thick_params, 'mu_normalized')

        df = p.df_from_named()
        df.to_csv('./out/ni_thick/ni_xmcd_{}.csv'.format(label))

        mid = int(3/5 *(len(p.get_checkpoint(3)[1])))
        Lz, Sz, Lz_guo, Sz_guo = sum_rules_basic_pipline(processor=p, mid_index=mid, n=n, l=l, c=c, n_3d=n_3d)
        sz.append(Sz)
        lz.append(Lz)
        szg.append(Sz_guo)
        lzg.append(Lz_guo)
        print(label)

        #cp_plotter.plot(p)

    print_sumrules_thick(lz, sz, lzg, szg)
    save_sumrules_csv(lz, sz, lzg, szg, 'ni_xmcd_thick', index_thick)



def ni_xmcd_strain():
    # ni xmcd strain dependence

    ni_strain_params = {'line_range': [835, 845],
                       'binary_filter': filter_circular,
                       'peak_1': (853, 854.5),
                       'peak_2': (871.48, 871.51),
                       'post': (882, 884.5),
                       'a': 1 / 3, 'delta': 1.5}

    # vary parameters individually
    #  tey      f1 ok,    f2,  f3,   f5 kaputt,     f4 ok
    a_list = [2/3-0.08, 2/3-0.19, 2/3-0.09, 2/3-0.1, 2/3]

    #  tfy     f1 bad, f2 looks good, f3 bad, f5 bad, f4 ok
    #a_list = [2/3-0.08, 2/3-0.07, 2/3-0.09, 2/3-0.1, 2/3+0.3]


    mid_point = 864 # eV

    n = 2
    l = 2
    c = 1
    n_3d = 8.2 # cluster model analysis reference 27 guo gupta

    lz = []
    sz = []
    lzg = []
    szg =[]

    measurements = table_xmcd_ni_high_b[:5]

    for i in range(0, len(measurements)):
        (a, b), label, _, _ = measurements[i]

        measurements_range = range(a + 1, b + 1)
        ni_strain_params['a'] = a_list[i]
        p = process_measurement(measurements_range, pipeline_basic_TEY_xmcd, ni_strain_params, 'mu_normalized')

        df = p.df_from_named()
        df.to_csv('./out/ni_strain/ni_xmcd_{}.csv'.format(label))

        mid = closest_idx(p.get_checkpoint(3)[0], mid_point) #int(22 / 35 * (len(p.get_checkpoint(3)[1])))
        print(label, mid, p.get_checkpoint(3)[1][mid])
        Lz, Sz, Lz_guo, Sz_guo = sum_rules_basic_pipline(processor=p, mid_index=mid, n=n, l=l, c=c, n_3d=n_3d)
        sz.append(Sz)
        lz.append(Lz)
        szg.append(Sz_guo)
        lzg.append(Lz_guo)
        print(label)

        #cp_plotter.plot(p)

    print_sumrules_strain(lz, sz, lzg, szg)
    save_sumrules_csv(lz, sz, lzg, szg, 'ni_xmcd_strain', index_strain)


def mn_xmcd_thickness():
    pipeline_TEY_xmcd = [Cut, SplitBy(binary_filter=filter_circular), Average, LineBG, FermiBG,
                               CheckPoint('left_right'),
                               CombineAverage, Normalize(save='mu0'), CheckPoint('normalized_xas'),
                               Integrate, CheckPoint('integral_xas'),
                               BackToNamed('left_right'), Normalize(to='mu0'), CombineDifference, CheckPoint('xmcd'),
                               Integrate, CheckPoint('integral_xmcd')]

    mn_thick_params = {'cut_range': (627, 662),
                       'line_range': (627, 635),
                       'binary_filter': filter_circular,
                       'peak_1': (642, 647),
                       'peak_2': (652, 656),
                       'post': (660, 662),
                       'a': 0.05, 'delta': 1.5 }

    n = 2
    l = 2
    c = 1
    mid_point = 649 # eV
    n_3d = 3.8 # guo gupta cluster analysis

    lz = []
    sz = []
    lzg = []
    szg = []

    measurements = table_xmcd_mn_high_b[-5:]

    for i in range(0, len(measurements)):
        (start, end), label, _, _ = measurements[i]
        measurements_range = range(start+1, end+1)

        p = process_measurement(measurements_range, pipeline_TEY_xmcd, mn_thick_params, 'mu_normalized')

        df = p.df_from_named()
        df.to_csv('./out/mn_thick/ni_xmcd_{}.csv'.format(label))

        mid = closest_idx(p.get_checkpoint(3)[0], mid_point) #int(22 / 35 * (len(p.get_checkpoint(3)[1])))
        Lz, Sz, Lz_guo, Sz_guo = sum_rules_basic_pipline(processor=p, mid_index=mid, n=n, l=l, c=c, n_3d=n_3d)
        sz.append(Sz)
        lz.append(Lz)
        szg.append(Sz_guo)
        lzg.append(Lz_guo)
        print(label)

        #cp_plotter.plot(p)

    print_sumrules_thick(lz, sz, lzg, szg)
    save_sumrules_csv(lz, sz, lzg, szg, 'mn_xmcd_thick', index_thick)


def main():
    ni_xmcd_thickness()
    ni_xmcd_strain()
    mn_xmcd_thickness()
    #mn_xmcd_thickness()

if __name__ == '__main__':
    main()
