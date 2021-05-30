from core import SingleMeasurementProcessor
import matplotlib.pyplot as plt


class CheckPointPlotter():
    def __init__(self):
        pass

    def plot(self, smp:SingleMeasurementProcessor):
        for i, cp in enumerate(smp.get_checkpoints()):
            plt.plot(smp.get_x(), cp, label=i)
        plt.show()

