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
            if isinstance(cp, list):
                ya, yb = cp
                plt.plot(smp.get_x(), ya, label=CheckpointPlotter.name(i) + ' filter a')
                plt.plot(smp.get_x(), yb, label=CheckpointPlotter.name(i) + ' filter b')
            elif len(cp.shape) == 1:
                plt.plot(smp.get_x(), cp, label=CheckpointPlotter.name(i))
            elif len(cp.shape) == 2:
                for j, y in enumerate(cp):
                    plt.plot(smp.get_x(), y, label=CheckpointPlotter.name(i, j))

        plt.legend()
        plt.show()

class CheckpointPlotterMulti():
    def __init__(self):
        pass

    @staticmethod
    def name(i, j=None):
        return f'Checkpoint {i}' if j is None else f'Checkpoint {i} - {j}'

    def plot(self, mmp:MultiMeasurementProcessor, labels=None):

        x = mmp.singles[0].get_x()

        for cps in mmp.get_checkpoints():

            for i, cp in enumerate(cps):
                if isinstance(cp, list):
                    ya, yb = cp
                    plt.plot(x, ya, label='a')
                    plt.plot(x, yb, label='b')
                elif len(cp.shape) == 1:
                    plt.plot(x, cp, label=CheckpointPlotter.name(i))
                elif len(cp.shape) == 2:
                    for j, y in enumerate(cp):
                        plt.plot(x, y, label=CheckpointPlotter.name(i, j))

        plt.legend()
        plt.show()

