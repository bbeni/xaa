#!/usr/bin/env python
"""Author: Beni
    -preprocess .dat file
    -save as pandas dataframe into dfs 
        (-> use function create_dfs)
"""
import argparse
import os.path
import re

import numpy as np
import pandas as pd

from .config import LABELS, PARAMS, CALCULATIONS, RENAMINGS

acquisition_regex = r'(#S.*?)#L\s+(.*?)(?:\Z|#C)'
number_date_regex = r'#S\s+(.*?)\s+.*?#D\s+(.*?)\n#C'

acquisition_regex = re.compile(acquisition_regex, re.M ^ re.DOTALL)
number_date_regex = re.compile(number_date_regex, re.M ^ re.DOTALL)


def valid_subset(a, b):
    '''takes lists. "a < b ?" '''
    for ai in a:
        if ai not in b:
            return False
    return True

def make_title(header):
    # find acquisition number and date
    m = re.match(number_date_regex, header)
    if m:
        return m.groups()

def make_df(header, text):
    x = text.splitlines()
    columns = x[0].split()
    if x[1:] != []:
        return pd.DataFrame(np.genfromtxt(x[1:], names=columns))

class ParamBox:
    def __init__(self, header):
        # find parameters
        param_strs = re.findall(r'#O\d+\s+(.*?)\n', header)
        self.p_labels = [s for line in param_strs for s in line.split()]

        # find parameter values
        param_values = re.findall(r'#P\d+\s+(.*?)\n', header)
        self.p_values = [float(s) for line in param_values for s in line.split()]

    def search(self, label):
        return self.p_values[self.p_labels.index(label)]

def create_dfs(boreas_file, index_range=(None, None), load_all_labels=False):
    '''returns dataframes, titles '''
    
    content = boreas_file.read()
    boreas_file.close()

    acquisitions = re.findall(acquisition_regex, content)

    dfs = []
    titles = []
    skips = []

    for header, text in acquisitions[index_range[0]:index_range[1]]:

        df = make_df(header, text)
        title = make_title(header)

        if df is None or title is None or not valid_subset([], df.columns):
            skips.append(title[0])
            continue

        if not load_all_labels:
            if LABELS != []:
                labels_valid = [label for label in LABELS if label in df]
                df = df[labels_valid]

        # update df from user defined PARAMS
        param_box = ParamBox(header)
        for p_label, p_index_label in PARAMS.items():
            df[p_label] = param_box.search(p_index_label)

        # update df from user defined CALCULATIONS
        for c_label, c in CALCULATIONS.items():
            f = c[0]
            args = c[1:]
            args = map(lambda label: df[label], args)
            df[c_label] = f(*args)

        # update df from user defined RENAMINGS
        if RENAMINGS != {}:
            for name, new_name in RENAMINGS.items():
                df = df.rename({name:new_name}, axis='columns')

        titles.append(title)
        dfs.append(df)

    print('skipped over {}'.format(skips))

    return dfs, titles


def save_acquisitions(dfs, titles, out_dir):
    '''store the dataframes as csvs'''

    print('Saved acquisitions: ')

    for i, df in enumerate(dfs):
        n, date = titles[i]
        path = out_dir + os.sep + "acquisition_{}.csv".format(n)
        df.to_csv(path, index=False, sep="\t")

        print("{}, ".format(n), end='')

    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('boreas_file', type=argparse.FileType('r'), help='boreas measurement file')
    parser.add_argument('--range', nargs=2, type=int, default=(0, None), help='range of measurements')
    parser.add_argument('--out_dir', default='csvs', type=str, help='csv output directory')

    args = parser.parse_args()
    a, b = args.range

    # silently create out_dir to save multiple csvs to
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    
    dfs, titles = create_dfs(args.boreas_file, index_range=(a,b))
    save_acquisitions(dfs, titles, args.out_dir)



