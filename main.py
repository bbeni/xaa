#!/usr/bin/env python3
import pandas as pd
import numpy as np

from core import SingleMeasurementProcessor
from plotting import CheckPointPlotter
from operations import Normalize, CheckPoint, LineBG, Back, Integrate



# another requirement is that we have estimated parameters in case nothing is provided

# we can structure it like a tree
# pipline examples:
#  single dfs -> select_x,y -> average -> normalize(save='m1')
#                           -> split_by(filter) -> average -> normalize(to='m1') -> difference -> integrate -> integral_gather
#  single dfs -> slect_x,y -> average -> bg_subtract(method) -> normalize -> vary('x', [1,2]) -> bg_subtract('fermi') -> integrate -> vary -> integral_gather


# file structure proposal
# main.py -> tests and examples
# single.py
# helpers.py
# multi.py
# preprocess.py
# plots.py

def test():

    pipeline = [Normalize, CheckPoint, LineBG, CheckPoint]
    p = SingleMeasurementProcessor()
    p.add_pipeline(pipeline)


    p.add_params()
    p.add_data()
    p.run()

    plotter = CheckPointPlotter()
    plotter.plot(p)

    #pipeline1 = [SelectXY, Interpolate, Average, Normalize(save='m1'), Back(steps=2), SplitBy(f=left), Average, Normalize(to='m1'), Difference, Integrate]

if __name__ == "__main__":
    test()
