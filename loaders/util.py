import os
import pandas as pd
from .preprocess import create_dfs
from typing import List, Union, Iterable

def _load_once(function):
    data_managers = {}
    def wrapper(*args):
        fname = args[0]
        if fname in data_managers:
            return data_managers[fname]
        else:
            dm = function(*args)
            data_managers[fname] = dm
            return dm
    return wrapper

@_load_once
def load_dm(filename):
    return DataManager(data_file=filename, load_cached=True)


def get_measurements_boreas_file(filename, indices: Union[List[int], List[List[int]]]) \
        -> Union[List[pd.DataFrame], List[List[pd.DataFrame]]]:
    """indides either a list of indices or a list[list of indices]"""
    if type(indices[0]) is range or type(indices[0]) is list:
        dfs = []
        for indices_sublist in indices:
           dfs.append(load_dm(filename).select(indices_sublist))
        return dfs
    else:
        return load_dm(filename).select(indices)


class DataManager:
    '''utility Class to manage measurement data'''

    def __init__(self, data_file, load_cached=False, load_all_labels=False):
        if load_cached and os.path.exists('.cached_data'):
            self.from_cached()
            return

        with open(data_file) as f:
            dfs, titles = create_dfs(f, load_all_labels=load_all_labels)

        index = [int(i) for i, _ in titles]
        self.dfs = dict(zip(index, dfs))

        if load_cached and not os.path.exists('.cached_data'):
            self.to_cached()

    def select_single(self, index, column_name=''):
        if column_name:
            return self.dfs[index][column_name].to_numpy()
        return self.dfs[index]

    def select(self, indices, column_name='', filter=None):
        selected = []
        for i in indices:
            selected.append(self.dfs[i])
        if filter:
            selected = [df for df in selected if filter(df)]
        if column_name:
            selected = [df[column_name].to_numpy() for df in selected]

        return selected

    def indices_filter(self, indices, filter):
        indices_filtered = [i for i in indices if filter(self.dfs[i])]
        return indices_filtered

    def clean_cached(self):
        # TODO implement
        pass

    def to_cached(self):
        os.mkdir('.cached_data')
        for i, df in self.dfs.items():
            df.to_pickle('.cached_data/{}.pkl'.format(i))

    def from_cached(self):
        self.dfs = {}
        for root, dirs, files in os.walk('.cached_data'):
            for filename in files:
                index = int(filename[:-4])
                self.dfs[index] = pd.read_pickle('.cached_data' + os.sep + filename)