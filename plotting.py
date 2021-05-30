from core import SingleMeasurementProcessor
import matplotlib.pyplot as plt


class CheckpointPlotter():
    def __init__(self):
        pass

    @staticmethod
    def name(i, j=None):
        return f'Checkpoint {i}' if j is None else f'Checkpoint {i} - {j}'

    def plot(self, smp:SingleMeasurementProcessor):
        for i, cp in enumerate(smp.get_checkpoints()):
            if len(cp.shape) == 1:
                plt.plot(smp.get_x(), cp, label=CheckpointPlotter.name(i))
            elif len(cp.shape) == 2:
                for j, y in enumerate(cp):
                    plt.plot(smp.get_x(), y, label=CheckpointPlotter.name(i, j))

        plt.legend()
        plt.show()

