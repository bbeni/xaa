# README #

xaa is a python module designed to help the experimental physicist to make processing of XAS data easy. This code is done as part of a bachelor thesis project by Benjamin Froelich at University of Zuerich.

## Requirements ##

* Python 3.6 +  
	The following python modules are required
	* numpy
	* pandas
	* scipy
	* matplotlib
	* lmfit
	
First install [anaconda](https://www.anaconda.com) ar [miniconda](https://conda.io). then install the dependencies with:

    conda install -c conda-forge lmfit
    conda install numpy scipy matplotlib pandas

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

* `Checkpoint` intermediate x and y data store to extract intermediate steps.
* `BackToNamed('a')` is used to jump back in the pipeline to a named `Checkpoint(name='a')`
* `BackTo(2)` jumps back to the second pipline operation state.
* `Back(3)` jumps back 3 pipline operations.


## Example ##





