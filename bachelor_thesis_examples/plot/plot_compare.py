import matplotlib.pyplot as plt
import pandas as pd
from xaa.plotting import OneAxisPlot


df_ni_thick = pd.read_csv('../out/compare_data/ni_xmcd_thick.csv', index_col=0)
df_ni_strain = pd.read_csv('../out/compare_data/ni_xmcd_strain.csv', index_col=0)
df_mn_thick = pd.read_csv('../out/compare_data/mn_xmcd_thick.csv', index_col=0)


# LAO, NGO, LSAT, LGO, STO
strain = [-2.190, -0.019, 0.188, 0.342, 1.133]


class ComparePlot(OneAxisPlot):
    def __init__(self, *args, **kwargs):
        kwargs["figsize"] = (3.54*2/3, 6/3)
        kwargs["axis_dimensions"] = [0.25, 0.2, 0.74, 0.74]
        super().__init__( *args, **kwargs)
        self.set_legend_layout(bbox_to_anchor=(0, 1), loc='upper left', frameon=True, fontsize=8)

plot = ComparePlot('Thickness [uc]', 'magnetic moment [$\mu_b$]')
plot.ylim((0, 2))
plot.plot(df_ni_thick.index, -df_ni_thick['Lz_guo'], 'L_z', 0, scatter=True)
plot.plot(df_ni_thick.index, -df_ni_thick['Sz_guo'], 'S_z', 1, scatter=True)
plot.finish(save="ni_thick_compare_sumrules.pdf")

plot = ComparePlot('Strain [uc]', 'magnetic moment [$\mu_b$]')
plot.ylim((0, 2))
plot.plot(strain, -df_ni_strain['Lz_guo'], 'L_z', 0, scatter=True)
plot.plot(strain, -df_ni_strain['Sz_guo'], 'S_z', 1, scatter=True)
plot.finish(save="ni_strain_compare_sumrules.pdf")

plot = ComparePlot('Thickness [uc]', 'magnetic moment [$\mu_b$]')
plot.ylim((0, 2))
plot.plot(df_mn_thick.index, -df_mn_thick['Lz_guo'], 'L_z', 0, scatter=True)
plot.plot(df_mn_thick.index, -df_mn_thick['Sz_guo'], 'S_z', 1, scatter=True)
plot.finish(save="mn_thick_compare_sumrules.pdf")


