import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import cm


class XMCDPlot():
    def __init__(self, xlabel, ylabel, figsize=(5.90551, 2*1.5), \
            axis_dimensions=[0.11, 0.15, 0.86, 0.83], \
            inset_axis_dimensions=[0.5,0.62,0.4,0.33], \
            color_map_name='tab10', n_colors=2):
        # from https://towardsdatascience.com/an-introduction-to-making-scientific-publication-plots-with-python-ea19dfa7f51e
        # Edit the font, font size, and axes width
        mpl.rcParams['font.family'] = 'Avenir'
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.linewidth'] = 1
        self.colors = cm.get_cmap(color_map_name, n_colors)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(axis_dimensions)
        ax_inset = fig.add_axes(inset_axis_dimensions)

        # Edit the major and minor ticks of the x and y axes
        ax.xaxis.set_tick_params(which='major', size=7, width=1, direction='in', top='on')
        ax.xaxis.set_tick_params(which='minor', size=5, width=0.7, direction='in', top='on')
        ax.yaxis.set_tick_params(which='major', size=7, width=1, direction='in', right='on')
        ax.yaxis.set_tick_params(which='minor', size=5, width=0.7, direction='in', right='on')
        self.fig = fig
        self.ax = ax
        self.ax_inset = ax_inset

        self.axis_naming(xlabel, ylabel)


    def plot(self, x, y, label, color_nr=-1, scatter=False):
        c = self.colors(color_nr) if color_nr >= 0 else 'k'
        if not scatter:
            self.ax.plot(x, y, label=label, color=c)
        else:
            self.ax.scatter(x, y, label=label, color=c)

    def plot_inset(self, thickness, int_xmcd, color_nr):
        c = self.colors(color_nr) if color_nr >= 0 else 'k'
        self.ax_inset.scatter(thickness, int_xmcd, color=c)

    def axis_naming(self, xlabel, ylabel):
        self.ax.set_xlabel(xlabel, labelpad=0)
        self.ax.set_ylabel(ylabel, labelpad=0)

    def inset_axis_naming(self, x, y):
        self.ax_inset.set_xlabel(x)
        self.ax_inset.set_ylabel(y)

    def xlim(self, xlim):
        self.ax.set_xlim(xlim)

    def ylim(self, ylim):
        self.ax.set_ylim(ylim)

    def finish(self, save=''):
        self.ax.legend(bbox_to_anchor=(0.05, 0.95), loc='upper left', frameon=False, fontsize=9)
        if not save:
            plt.show()
        else:
            plt.savefig(save, dpi=300)


class IntegralPlotXMCD:
    def __init__(self, xlabel, ylabel, figsize=(5.90551/2*3, 1.5*3), \
            axis_dimensions=[0.11, 0.15, 0.86, 0.83], \
            inset_axis_dimensions=[0.5,0.62,0.4,0.33], \
            color_map_name='tab10', n_colors=2):
        # from https://towardsdatascience.com/an-introduction-to-making-scientific-publication-plots-with-python-ea19dfa7f51e
        # Edit the font, font size, and axes width
        mpl.rcParams['font.family'] = 'Avenir'
        plt.rcParams['font.size'] = 11
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

    def plot(self, x, y, label, color_nr=-1, scatter=False):
        c = self.colors(color_nr) if color_nr >= 0 else 'k'
        if not scatter:
            self.ax.plot(x, y, label=label, color=c)
        else:
            self.ax.scatter(x, y, label=label, color=c)

    def axis_naming(self, xlabel, ylabel):
        self.ax.set_xlabel(xlabel, labelpad=0)
        self.ax.set_ylabel(ylabel, labelpad=0)

    def xlim(self, xlim):
        self.ax.set_xlim(xlim)

    def ylim(self, ylim):
        self.ax.set_ylim(ylim)

    def finish(self, save=''):
        self.ax.legend(bbox_to_anchor=(0.05, 0.95), loc='upper left', frameon=False, fontsize=9)
        if not save:
            plt.show()
        else:
            plt.savefig(save, dpi=300)


class XASPlot():
    def __init__(self, xlabel, ylabel, figsize=(5.90551/2*2, 1.5*2), \
            axis_dimensions=[0.11, 0.15, 0.86, 0.83], \
            inset_axis_dimensions=[0.5,0.62,0.4,0.33], \
            color_map_name='tab10', n_colors=2):
        # from https://towardsdatascience.com/an-introduction-to-making-scientific-publication-plots-with-python-ea19dfa7f51e
        # Edit the font, font size, and axes width
        mpl.rcParams['font.family'] = 'Avenir'
        plt.rcParams['font.size'] = 11
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

    def plot(self, x, y, label, color_nr=-1, scatter=False):
        c = self.colors(color_nr) if color_nr >= 0 else 'k'
        if not scatter:
            self.ax.plot(x, y, label=label, color=c)
        else:
            self.ax.scatter(x, y, label=label, color=c)

    def axis_naming(self, xlabel, ylabel):
        self.ax.set_xlabel(xlabel, labelpad=0)
        self.ax.set_ylabel(ylabel, labelpad=0)

    def xlim(self, xlim):
        self.ax.set_xlim(xlim)

    def ylim(self, ylim):
        self.ax.set_ylim(ylim)

    def finish(self, save=''):
        self.ax.legend(bbox_to_anchor=(0.05, 0.95), loc='upper left', frameon=False, fontsize=9)
        if not save:
            plt.show()
        else:
            plt.savefig(save, dpi=300)

if __name__ == "__main__":
    plot = IntegralPlotXMCD('x', 'y')
    plot.plot([], [], label='test')
    plot.finish()

    #plot = XMCDPlot('x', 'y')
    #plot.plot([], [], label='test')
    #plot.finish()