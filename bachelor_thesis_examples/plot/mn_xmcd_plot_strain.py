import pandas as pd
from plot_templates import XMCDPlot

csv_dir = '../out/ni_thick/'
savedir = '../../../bachelor-thesis/figures/'

# labels_name = ['LAO', 'NGO', 'LSAT', 'LGO', 'STO', '3uc', '6uc', '10uc', '21uc']
# labels_id = ['f1', 'f2', 'f3', 'f5', 'f4', 'b2', 'b1', 'b3', 'f6']
sample_ids_strain = ['f1', 'f2', 'f3', 'f5', 'f4']
labels_strain = ['LAO', 'NGO', 'LSAT', 'LGO', 'STO']
strain = [-2.190, -0.019, 0.188, 0.342, 1.133]
sample_ids_thickness = ['b1', 'b2', 'b3', 'f4', 'f6']
thickness = [6, 3, 10, 30, 21]