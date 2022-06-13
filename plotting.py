from core import SingleMeasurementProcessor, MultiMeasurementProcessor
import matplotlib.pyplot as plt


class CheckpointPlotter():
    def __init__(self):
        pass

    @staticmethod
    def name(i, j=None):
        return f'Checkpoint {i}' if j is None else f'Checkpoint {i} - {j}'

    def plot(self, smp:SingleMeasurementProcessor):
        for i, cp in enumerate(smp.get_checkpoints()):
            if len(cp) == 3:
                x, ya, yb = cp
                plt.plot(x, ya, label=CheckpointPlotter.name(i) + ' filter a')
                plt.plot(x, yb, label=CheckpointPlotter.name(i) + ' filter b')
            elif len(cp[1].shape) == 1:
                x, y = cp
                plt.plot(x, y, label=CheckpointPlotter.name(i))
            elif len(cp[1].shape) == 2:
                x, ys = cp
                for j, y in enumerate(ys):
                    plt.plot(x, y, label=CheckpointPlotter.name(i, j))

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

