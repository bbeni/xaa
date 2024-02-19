# README #

xaa is a python module designed to help the experimental physicist to make processing of XAS data easy. This code is done as part of a bachelor thesis project by Benjamin Froelich at University of Zuerich.

## Installation ##

You need python 3.6+, for example install it with [anaconda](https://www.anaconda.com) or [miniconda](https://conda.io). (Optionally) create a conda environement using conda/anaconda and activate it.

	conda create -n test_env python=3
	conda activate test_env

Install the dependencies with:

    conda install -c conda-forge lmfit
    conda install numpy scipy matplotlib pandas

Go to the directory where the setup.py lives and run:

	pip install .

You should now be able to `import xaa`.

## XMCD Example ##

A small example to analyse XMCD is given. We first need to import the modules.

```python
import pandas as pd
from xaa.core import SingleMeasurementProcessor
from xaa.plotting import CheckpointPlotter
from xaa.operations import Normalize, CheckPoint, Integrate, \
    SplitBy, Average, CombineDifference, CombineAverage, BackToNamed

from xaa.operations import LineBGRemove
```

Let's assume we have 8 measurements, 4 for left and 4 for right polarized xrays with a column indicating the circular polarization (example -1 or 1, should be the same for one Dataframe!). One Dataframe could look like this:

| energy | XAS   | polarization |
|--------|-------|--------------|
| 630.0  | 0.0   | -1           |
| 630.1  | 0.02  | -1           |
| 630.2  | 0.034 | -1           |
| ...    | ...   | ...          |

Assuming the filenames are 'm1.csv', 'm2.csv', ... , 'm8.csv' that hold the 8 measurements. We load them as pandas dataframes. We also need to define a pipeline and a measurement processor `p`. And also a few parameters needed for some Operations and a function that determines the polarization. The `CheckPointPlotter` is used as a quick way to investigate the data at different `CheckPoint`s. When we are happy we can extract the processed data as a csv that holds all `CheckPoint`s with `p.df_from_named()`.

```python
measurements = [pd.read_csv('m{}.csv'.format(i)) for i in range(1,9)]

def is_left_polarized(df):
    return df.polarization[0] > 0

pipeline = [SplitBy(binary_filter=is_left_polarized), Average, CheckPoint('raw'), 
            LineBGRemove,  CheckPoint('line_removed'),
            CombineAverage, Normalize(save='mu0'), CheckPoint('normalized_xas'),
            Integrate, CheckPoint('integral_xas'),
            BackToNamed('line_removed'), Normalize(to='mu0'), CombineDifference, CheckPoint('xmcd'),
            Integrate, CheckPoint('integral_xmcd')]

pipeline_params = {'line_range': [630, 635]} # we need to specify this because of the LineBgRemove Operation.

p = SingleMeasurementProcessor()
p.add_pipeline(pipeline)
p.add_params(pipeline_params)
p.check_missing_params()
p.add_data(dataframes, x_column='energy', y_column='XAS')
p.run()

# now the operations are applied to the data and we can retrieve any checkpoint with
print(p.get_checkpoint('xmcd'))

# we can also save everything to a csv
df = p.df_from_named()
df.to_csv('test_run.csv')

# or plot the results to see what happend for debugging
# give it a list of checkpoint names to plot.
plotter = CheckPointPlotter(['normalized_xas', 'xmcd'])
plotter.plot(p)

```

There are more Operations to Remove the background signal. LineBGRemove, FermiBG, FermiWallAdaptive, ArctanAdaptiveRemove, FermiWallAdaptive2, FermiBGfittedA. These all expect some parameters that are data specific. the `p.check_missing_params()` will show what is needed.

## Working principle ##

Input Data is assumed to consist of a 1 dimensional x values array (usally X-Ray energy values) and y values data which can be 1 or 2 dimensional.

A pipepline is defined as a list of `Operation`s, that will be applied in order to manipulate the data.

Operations that change the data include:

* `Average` can only be applied to 2 d y data. averages multiple y measurements.
* `Flip` maps y &rarr; -y
* `Add` maps y &rarr; y + c
* `Normalize(save='n1')` normalizes y to 1 and stores the normalization constant named 'n1'
* `Normalize(to='n1')` mormalized y to a previously stored constant named 'n1'
* `ApplyFunctionToY(function=f)` applies a defined function `f(y)` to y &rarr; f(y)
* `SplitBy(binary_filter=filter)` expects a filter that takes a pd.Dataframe and returns True or False. Splits 2d y data into yA and yB based on some value of the dataframe. Used for XMCD and XLD as example. If afterwards Normalize or Average is applied, it does it for yA and yB individually.
* `CombineDifference` maps 1d split data yA, yB &rarr; yB - yA
* `CombineAverage` maps 1d split data yA, yB &rarr; (yA + yB)/2

Higher level manipulations include:

* `Integrate` does a numerical cumulative integral from x_min to x_max of y.
* `LineBG(line_range)` subtracts a line ax + b fitted to line_range
* `FermiBg` subtracts a fermi type double step background. expected_params = ['peak_1', 'peak_2', 'post', 'delta', 'a']

Operations that manipulate the pipeline include:

* `Checkpoint('a')` intermediate x and y data store with name 'a'.
* `BackToNamed('a')` is used to jump back in the pipeline to `Checkpoint('a')`
* `BackTo(2)` jumps back to the second pipline operation state.
* `Back(3)` jumps back 3 pipline operations.


## Help ##

If you need some help, have good ideas how to improve it or find bugs please let me know :)
