import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
from .core import SingleMeasurementProcessor, MultiMeasurementProcessor

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

class CheckpointPlotter:
    def __init__(self, named_cps_to_plot=[]):
        self.named_cps_to_plot = named_cps_to_plot

    @staticmethod
    def name(name:str, j=None):
        return f'{name}' if j is None else f'{name} - {j}'

    def plot(self, smp:SingleMeasurementProcessor, xlabel='x', ylabel='y', legend=True):
        plt.figure(dpi=250)
        for i, (name, cp) in enumerate(smp.get_checkpoints().items()):
            if self.named_cps_to_plot and not name in self.named_cps_to_plot:
                continue

            if len(cp) == 3:
                x, ya, yb = cp

                if len(ya.shape) > 1:
                    for y in ya:
                        plt.plot(x, y, label=CheckpointPlotter.name(name) + ' filter a')
                else:
                    plt.plot(x, ya, label=CheckpointPlotter.name(name) + ' filter a')

                if len(yb.shape) > 1:
                    for y in yb:
                        plt.plot(x, y, label=CheckpointPlotter.name(name) + ' filter a')
                else:
                    plt.plot(x, yb, label=CheckpointPlotter.name(name) + ' filter b')

            elif len(cp[1].shape) == 1:
                x, y = cp
                plt.plot(x, y, label=CheckpointPlotter.name(name))
            elif len(cp[1].shape) == 2:
                x, ys = cp
                for j, y in enumerate(ys):
                    plt.plot(x, y, label=CheckpointPlotter.name(name, j))

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if legend:
            plt.legend()
        plt.show()

class CheckpointPlotterMulti():
    def __init__(self):
        pass

    @staticmethod
    def name(i, j=None):
        return f'Checkpoint {i}' if j is None else f'Checkpoint {i} - {j}'

    def plot(self, mmp:MultiMeasurementProcessor, labels=None):

        for cps in mmp.get_checkpoints():
            for i, cp in enumerate(cps):
                if len(cp) == 3:
                    x, ya, yb = cp
                    plt.plot(x, ya, label='a')
                    plt.plot(x, yb, label='b')
                elif len(cp[1].shape) == 1:
                    x, y = cp
                    plt.plot(x, y, label=CheckpointPlotter.name(i))
                elif len(cp[1].shape) == 2:
                    x, ys = cp
                    for j, y in enumerate(ys):
                        plt.plot(x, y, label=CheckpointPlotter.name(i, j))

        plt.legend()
        plt.show()



class TFYTEYComparePlot():
    """Plot 3 xas curves labeled with tey tfy 1/tfy, last CheckPoint"""
    def __init__(self):
        pass

    def plot_xy(self, x1, x2, y1, y2):
        # from https://towardsdatascience.com/an-introduction-to-making-scientific-publication-plots-with-python-ea19dfa7f51e
        # Edit the font, font size, and axes width
        mpl.rcParams['font.family'] = 'Avenir'
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.linewidth'] = 1
        colors = cm.get_cmap('tab10', 3)

        fig = plt.figure(figsize=(3.54*1.5, 2*1.5))
        ax = fig.add_axes([0.15, 0.2, 0.84, 0.79])

        # Edit the major and minor ticks of the x and y axes
        ax.xaxis.set_tick_params(which='major', size=10, width=1, direction='in', top='on')
        ax.xaxis.set_tick_params(which='minor', size=7, width=1, direction='in', top='on')
        ax.yaxis.set_tick_params(which='major', size=10, width=1, direction='in', right='on')
        ax.yaxis.set_tick_params(which='minor', size=7, width=1, direction='in', right='on')

        ax.plot(x1, y1, label='Total Electron Yield', color=colors(0))
        ax.plot(x2, y2, label='Total Fluorescence Yield', color=colors(1))
        #ax.plot(x3, y3, label='1/(Fluorescence Yield))', color=colors(2))

        ax.set_xlabel('Photon Energy [eV]', labelpad=0)
        ax.set_ylabel('Normalized $\mu$ (arb.units)', labelpad=0)

        self.ax = ax

        #plt.show()

        #plt.savefig('tfyvstey.pdf', dpi=300)

    def plot_square(self, ll, ur, label):
        plt.plot([ll[0], ur[0], ur[0], ll[0], ll[0]], [ll[1], ll[1], ur[1], ur[1], ll[1]], label=label)

    def finish(self, fname=None):
        self.ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=11)
        if fname:
            plt.savefig(fname, dpi=300)
        else:
            plt.show()

    def plot_old(self, smpTEY:SingleMeasurementProcessor, smpTFY:SingleMeasurementProcessor):
        # modified example from https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
        x1, y1 = smpTEY.get_checkpoint(-1)
        x2, y2 = smpTFY.get_checkpoint(-1)

        # Plot Line1 (Left Y Axis)
        fig, ax1 = plt.subplots(1, 1, figsize=(3.54, 2), dpi=300)
        ax1.plot(x1, y1, color='tab:red')

        # Plot Line2 (Right Y Axis)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(x2, y2, color='tab:blue')

        # Decorations
        # ax1 (left Y axis)
        ax1.set_xlabel('Photon Energy [eV]', fontsize=11)
        ax1.tick_params(axis='x', rotation=0, labelsize=11)
        ax1.set_ylabel('Total Electron Yield [arb. units]', color='tab:red', fontsize=11)
        ax1.tick_params(axis='y', rotation=0, labelcolor='tab:red')
        ax1.grid(alpha=.4)

        # ax2 (right Y axis)
        ax2.set_ylabel("Total Fluoresence Yield [arb. units]", color='tab:blue', fontsize=11)
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        #ax2.set_xticks(np.arange(0, len(x), 60))
        #ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize': 10})
        ax2.set_title("TFY and TEY raw Measurement at Mn L3,2 Edge", fontsize=12)
        fig.tight_layout()
        plt.show()


class OneAxisPlot():
    def __init__(self, xlabel, ylabel, figsize=(3.54, 2), axis_dimensions=[0.15, 0.20, 0.84, 0.77], color_map_name='tab10', n_colors=2):
        # from https://towardsdatascience.com/an-introduction-to-making-scientific-publication-plots-with-python-ea19dfa7f51e
        # Edit the font, font size, and axes width
        mpl.rcParams['font.family'] = 'Avenir'

        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1
        self.colors = cm.get_cmap(color_map_name, n_colors)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(axis_dimensions)

        # Edit the major and minor ticks of the x and y axes
        ax.xaxis.set_tick_params(which='major', size=7, width=1, direction='in', top='on')
        ax.xaxis.set_tick_params(which='minor', size=5, width=0.7, direction='in', top='on')
        ax.yaxis.set_tick_params(which='major', size=7, width=1, direction='in', right='on')
        ax.yaxis.set_tick_params(which='minor', size=5, width=0.7, direction='in', right='on')
        self.fig = fig
        self.ax = ax

        self.axis_naming(xlabel, ylabel)

        self.legend_kwargs = None

    def plot(self, x, y, label, color_nr=-1, scatter=False):
        c = self.colors(color_nr) if color_nr >= 0 else 'k'
        if not scatter:
            self.ax.plot(x, y, label=label, color=c)
        else:
            self.ax.scatter(x, y, label=label, color=c)

    def errorbar(self, x, y, y_err, label, color_nr=-1, scatter=False, fmt='o'):
        c = self.colors(color_nr) if color_nr >= 0 else 'k'
        if not scatter:
            self.ax.errorbar(x, y, yerr=y_err*2, label=label, color=c)
        else:
            self.ax.errorbar(x, y, yerr=y_err*2, fmt=fmt, markersize=5, capsize=0, label=label, color=c)

    def axis_naming(self, xlabel, ylabel):
        self.ax.set_xlabel(xlabel, labelpad=0)
        self.ax.set_ylabel(ylabel, labelpad=0)

    def xlim(self, xlim):
        plt.xlim(xlim)

    def ylim(self, ylim):
        plt.ylim(ylim)


    def set_legend_layout(self, **kwargs):
        self.legend_kwargs = kwargs

    def finish(self, save=''):
        if self.legend_kwargs:
            self.ax.legend(**self.legend_kwargs)
        else:
            self.ax.legend(bbox_to_anchor=(1, 0), loc='lower right', frameon=False, fontsize=9)

        self.fig.tight_layout()
        if not save:
            plt.show()
        else:
            plt.savefig(save, dpi=300)#,  bbox_inches="tight")