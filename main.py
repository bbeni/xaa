#!/usr/bin/env python3
import pandas as pd
import numpy as np



# TODO implement select_x_y, average, normalize_save(variable='m1'), normalize_to('m1'),
#  bg_subtract, split_by, difference, vary,

class Single:
    def __init__(self, dfs):
        self.dfs = dfs


# should be easy to implement once Single works
class Multi:
    def __init__(self, dfs_list):
        self.singles = []
        for dfs in dfs_list:
            self.singles.append(Single(dfs))


# should be able to extract needed data from pipline
class ExamplePlot:
    pass


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



if __name__ == "__main__":
    pass
