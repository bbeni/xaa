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

from plot.plot_templates import XASPlot

savedir = '../../bachelor-thesis/figures/'

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


# room temp 0 b field
## range, id, temp, bfield
table_ni_xld = [
    [(87,   91), 'f1', 300, 0],
    [(92,   96), 'f2', 300, 0],
    [(97,  101), 'f3', 300, 0],
    [(107, 111), 'f5', 300, 0],
    [(102, 106), 'f4', 300, 0],
    [(122, 126), 'b2', 300, 0],
    [(117, 121), 'b1', 300, 0],
    [(127, 131), 'b3', 300, 0],
    [(112, 116), 'f6', 300, 0]]

table_mn_xld = [
    [(42,  46), 'f1', 300, 0],
    [(47,  51), 'f2', 300, 0],
    [(52,  56), 'f3', 300, 0],
    [(62,  66), 'f5', 300, 0],
    [(57,  61), 'f4', 300, 0],
    [(77,  81), 'b2', 300, 0],
    [(72,  76), 'b1', 300, 0],
    [(82,  86), 'b3', 300, 0],
    [(67,  71), 'f6', 300, 0],]


cp_plotter = CheckpointPlotter()

filter_linear = lambda df: df.polarization[0] > np.pi/4

pipeline_basic_xas = [SplitBy(filter_linear), Average, CombineAverage, LineBG, Normalize, CheckPoint('xas')]

def process_measurement(measurements_range, pipeline, pipeline_params, y_column):
    dataframes = get_measurements_boreas_file('data_files/SH1_Tue09.dat', measurements_range)

    p = SingleMeasurementProcessor()
    p.add_pipeline(pipeline)

    p.add_params(pipeline_params)
    p.check_missing_params()
    p.add_data(dataframes, x_column='energy', y_column=y_column)
    p.run()
    return p


index_strain = ['LAO', 'NGO', 'LSAT', 'LGO', 'STO']
strain = [-2.190, -0.019, 0.188, 0.342, 1.133]
index_thick = [30, 3, 6, 10, 21] # uc

def ni_xas_thickness():
    ni_thick_params = {#'cut_range': [835, 877],
                       'line_range': [835, 845],
                       'binary_filter': filter_linear,
                       'peak_1': (853, 854.5),
                       'peak_2': (871.48, 871.51),
                       'post': (875, 884.5),
                       'a': 2 / 3, 'delta': 1.5}

    l2 = 868, 875
    l3 = 850, 861

    ys = []

    xas_plot = XASPlot('Photon Energy [eV]', 'XAS [arb.units]', n_colors=5)

    i = 0
    for (a, b), label, _, _ in table_ni_xld[-4:] + [table_ni_xld[-5]]:
        measurements_range = range(a+1, b+1)
        p = process_measurement(measurements_range, pipeline_basic_xas, ni_thick_params, 'mu_normalized')

        x = p.get_checkpoint(0)[0]
        y = p.get_checkpoint(0)[1]
        xas_plot.plot(x, y+i*0.025, label="{} u.c.".format((index_thick[-4:] + [index_thick[-5]])[i]), color_nr=i)
        i += 1

    xas_plot.xlim((845, 880))
    xas_plot.finish(save=savedir+'ni_xas_compare.pdf')

def mn_xas_thickness():
    mn_thick_params = {'cut_range': (627, 662),
                       'line_range': (627, 635),
                       'binary_filter': filter_linear,
                       'peak_1': (642, 647),
                       'peak_2': (652, 656),
                       'post': (660, 662),
                       'a': 0.05, 'delta': 1.5 }

    xas_plot = XASPlot('Photon Energy [eV]', 'XAS [arb.units]', n_colors=5)

    i = 0
    for (a, b), label, _, _ in table_mn_xld[-4:] + [table_mn_xld[-5]]:
        measurements_range = range(a + 1, b + 1)
        p = process_measurement(measurements_range, pipeline_basic_xas, mn_thick_params, 'mu_normalized')

        x = p.get_checkpoint(0)[0]
        y = p.get_checkpoint(0)[1]
        xas_plot.plot(x, y, label="{} u.c.".format((index_thick[-4:] + [index_thick[-5]])[i]), color_nr=i)
        i += 1

    xas_plot.xlim((635,660))
    xas_plot.finish(save=savedir+'mn_xas_compare.pdf')



def ni_xas_strain():
    ni_thick_params = {#'cut_range': [835, 877],
                       'line_range': [835, 845],
                       'binary_filter': filter_linear,
                       'peak_1': (853, 854.5),
                       'peak_2': (871.48, 871.51),
                       'post': (875, 884.5),
                       'a': 2 / 3, 'delta': 1.5}

    xas_plot = XASPlot('Photon Energy [eV]', 'XAS [arb.units]', n_colors=5)

    i = 0
    for (a, b), label, _, _ in table_ni_xld[:5]:
        measurements_range = range(a+1, b+1)
        p = process_measurement(measurements_range, pipeline_basic_xas, ni_thick_params, 'mu_normalized')

        x = p.get_checkpoint(0)[0]
        y = p.get_checkpoint(0)[1]
        xas_plot.plot(x, y+0.025*i, label='{}% ({})'.format(strain[i],index_strain[i]), color_nr=i)
        i += 1

    xas_plot.xlim((845, 880))
    xas_plot.finish(save=savedir+'ni_xas_compare_strain.pdf')

def mn_xas_strain():
    mn_thick_params = {'cut_range': (627, 662),
                       'line_range': (627, 635),
                       'binary_filter': filter_linear,
                       'peak_1': (642, 647),
                       'peak_2': (652, 656),
                       'post': (660, 662),
                       'a': 0.05, 'delta': 1.5 }

    xas_plot = XASPlot('Photon Energy [eV]', 'XAS [arb.units]', n_colors=5)

    i = 0
    for (a, b), label, _, _ in table_mn_xld[:5]:
        measurements_range = range(a + 1, b + 1)
        p = process_measurement(measurements_range, pipeline_basic_xas, mn_thick_params, 'mu_normalized')

        x = p.get_checkpoint(0)[0]
        y = p.get_checkpoint(0)[1]
        xas_plot.plot(x, y, label='{}% ({})'.format(strain[i],index_strain[i]), color_nr=i)
        i += 1

    xas_plot.xlim((635,660))
    xas_plot.finish(save=savedir+'mn_xas_compare_strain.pdf')




def main():
    ni_xas_thickness()
    mn_xas_thickness()
    ni_xas_strain()
    mn_xas_strain()

if __name__ == '__main__':
    main()
