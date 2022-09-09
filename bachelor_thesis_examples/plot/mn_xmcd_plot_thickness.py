import pandas as pd
from plot_templates import XMCDPlot, IntegralPlotXMCD

csv_dir = '../out/mn_thick/'
savedir = '../../../bachelor-thesis/figures/'
appendix_savedir = savedir + 'appendix_plots/'


# labels_name = ['LAO', 'NGO', 'LSAT', 'LGO', 'STO', '3uc', '6uc', '10uc', '21uc']
# labels_id = ['f1', 'f2', 'f3', 'f5', 'f4', 'b2', 'b1', 'b3', 'f6']
sample_ids_strain = ['f1', 'f2', 'f3', 'f5', 'f4']
labels_strain = ['LAO', 'NGO', 'LSAT', 'LGO', 'STO']
strain = [-2.190, -0.019, 0.188, 0.342, 1.133]
sample_ids_thickness = ['b1', 'b2', 'b3', 'f4', 'f6']
thickness = [6, 3, 10, 30, 21]

dfs = []
for sid in sample_ids_thickness:
    dfs.append(pd.read_csv(csv_dir + '{}.csv'.format(sid), index_col=0))


def main():
    plot = XMCDPlot('Photon Energy [eV]', 'normalized xmcd [arb.u.]', n_colors=5)
    plot.xlim((635, 661))
    for i in [1, 0, 2, 4, 3]:
        plot.plot(dfs[i].index, dfs[i]['xmcd'], thickness[i], i)
        plot.plot_inset([thickness[i]], dfs[i]['integral_xmcd'].iloc[-1], i)
    plot.finish()#save=savedir + 'mn_thick_xmcd.pdf')

def side():
    for i in [1, 0, 2, 4, 3]:
        x = dfs[i].index
        plot = IntegralPlotXMCD('Photon Energy [eV]', 'normalized xmcd [arb.u.]',
                                figsize=(5.906/2, 1.9), n_colors=4)
        print('{} {} {}'.format(sample_ids_thickness[i], sample_ids_thickness[i], thickness[i]))
        plot.plot(x, dfs[i]['xmcd'], 'xmcd', 0)
        plot.plot(x, dfs[i]['normalized_xas'], 'xas', 1)
        plot.plot(x, dfs[i]['integral_xmcd'], 'integral xmcd', 2)
        plot.plot(x, dfs[i]['integral_xas'], 'integral xas', 3)
        plot.finish(save=appendix_savedir + 'mn_thick_xmcd' + str(thickness[i]) + '.pdf')

if __name__ == '__main__':
    #main()
    side()

