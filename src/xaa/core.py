from copy import deepcopy
import sys
import pandas as pd
import numpy as np
from typing import Union, List, Iterator

from .operations import Operation, PipelineOperation, TransformOperation, TransformOperationXY, SplitOperation, CombineOperation, CollapseOperation, Cut
from .operations import Back, BackTo, CheckPoint, BackToNamed

from .helpers import common_bounds, interpolate, StoragePool


class XYBlock:
    """
    Main Data Block. Only used internally.

    Stores an x and a y array that are the same dimensions. It is the main object to store
    all pipline steps.
    """
    def __init__(self, x, y, name=''):
        self.x = x
        self.y = y
        self.name = name

    def level(self):
        return len(self.y.shape)

    def apply_transform(self, transform: TransformOperation):
        if self.level() == 1:
            self.y = transform.do(self.x, self.y)
        elif self.level() == 2:
            for i in range(self.y.shape[0]):
                self.y[i,:] = transform.do(self.x, self.y[i,:])
        else:
            raise NotImplementedError('apply transform level ' + self.level())

    def apply_transform_xy(self, transform: TransformOperationXY):
        if self.level() == 1:
            self.x, self.y = transform.do(self.x, self.y)
        elif self.level() == 2:
            x, new_y = transform.do(self.x, self.y[0, :])
            ny = np.zeros((self.y.shape[0], new_y.shape[0]))
            ny[0,:] = new_y
            for i in range(1, self.y.shape[0]):
                _, ny[i,:] = transform.do(self.x, self.y[i,:])
            self.y = ny
            self.x = x
            print(self.x.shape, self.y.shape)
        else:
            raise NotImplementedError('level ' + self.level())

    def apply_collapse(self, collapse: CollapseOperation):
        self.y = collapse.do(self.x, self.y)

class XYBlockSplit:
    """XYBlock gets transformed to XYBlock split, after a Filter Operation."""
    def __init__(self, x, ya, yb, name=''):
        self.x = x
        self.ya = ya
        self.yb = yb
        self.name = name

    def level(self):
        return len(self.ya.shape)

    def apply_transform(self, transform: TransformOperation):
        if self.level() == 1:
            self.ya = transform.do(self.x, self.ya)
            self.yb = transform.do(self.x, self.yb)
        elif self.level() == 2:
            for i in range(self.ya.shape[0]):
                self.ya[i, :] = transform.do(self.x, self.ya[i,:])
            for i in range(self.yb.shape[0]):
                self.yb[i, :] = transform.do(self.x, self.yb[i,:])
        else:
            raise NotImplementedError('level ' + self.level())

    def apply_collapse(self, collapse: CollapseOperation):
        self.ya = collapse.do(self.x, self.ya)
        self.yb = collapse.do(self.x, self.yb)

class SingleMeasurementProcessor:
    """
    Main object to orchestrate all operations and all data.

    This is where all the magic happens.
    TODO: make easy example

    Methods
    -------
    add_data(dfs:[pd.Dataframe, .. ], x_colum="x", y_column="y")
        feed it with XAS data from a pandas DataFrame.
    add_pipline(pipline=[Average, CheckPoint("test"), .. ])
        add a processing pipeline.
    add_params({"a":123})
        add parameters if the pipeline Operations need them.
    check_missing_params()
        checks for missing params and prints their names.
    run()
        runs all operations in the pipline to the added data.
    get_named_checkpoint(name="test")
        returns the x and y data from that checkpoint.
    """
    def __init__(self):
        self.processed = False
        self.dfs = None

        self.xy_blocks = []

        self.override_params = False
        self.params = {}
        self.pipeline = []

        self.y_checkpoints = []
        self.y_checkpoints_names = []
        self.named_checkpoints = {}

        self.name_to_index = {}

        self.storage_pool = StoragePool()

    def get_x(self):
        return self.xy_blocks[0].x

    def add_data(self, dfs: Union[pd.DataFrame, List[pd.DataFrame]], x_column: str, y_column: str, n_samples: int = 2000):
        """add dataframes, then extract x and y columns, adjust x bounds, resample x and y values"""
        self.dfs = dfs if type(dfs) is list else [dfs]

        # resample x and y
        xlist = [df[x_column].to_numpy() for df in self.dfs]
        ylist = [df[y_column].to_numpy() for df in self.dfs]

        xa, xb = common_bounds(xlist)
        first_x = np.linspace(xa, xb, n_samples)
        first_y = np.zeros((len(self.dfs), n_samples))
        for i in range(len(dfs)):
            x, y = xlist[i], ylist[i]
            y_sampled = interpolate(x, y, first_x)
            first_y[i, :] = y_sampled

        first_block = XYBlock(first_x, first_y, name='Base Block')
        self.xy_blocks.append(first_block)

    def check_missing_params(self):
        failed = False
        if self.pipeline == []:
            raise ValueError('no pipeline added. supply a pipeline with the add_pipeline method first!')

        for i, operation in enumerate(self.pipeline):
            if isinstance(operation, Operation):
                missing = operation.missing_params({**self.params, **operation.params})
                if missing:
                    print(f'Missing Parameter(s) in {operation} pos[{i}]')
                    print(missing)
                    failed = True
            else:
                missing = operation.missing_params(self.params)
                if missing:
                    print(f'Missing Parameter(s) in {operation} pos[{i}]')
                    print('Operation not instanciated yet. looked only in global parameters...')
                    print(missing)
                    failed = True
        if not failed:
            #print('No Parameter is missing. Nice!')
            pass
        else:
            sys.exit(1)

    def add_params(self, params: dict, override: bool=False):
        self.override_params = override
        self.params = params

    def add_pipeline(self, pipeline: List[Operation]):
        for operation in pipeline:
            if isinstance(operation, type):
                if not issubclass(operation, Operation):
                    raise ValueError(f'{operation} is a class but not a subclass of Operation')
            elif isinstance(operation, Operation):
                pass
            else:
                raise ValueError(f'{operation} is an object is not an instance of Operation')

            self.pipeline.append(operation)

    def _instantiate_operations_and_apply_params(self):
        for i in range(len(self.pipeline)):
            operation = self.pipeline[i]
            if isinstance(operation, type):
                if operation.expected_params == []:
                    self.pipeline[i] = operation()
                    continue
                # find global params and instantiate
                p = {k:v for k,v in self.params.items() if k in operation.expected_params}
                self.pipeline[i] = operation(**p)
            elif isinstance(operation, Operation):
                if operation.check_params(operation.params):
                    # TODO override functionality
                    continue
                elif operation.check_params({**self.params, **operation.params}):
                    p = {k: v for k, v in self.params.items() if k in operation.expected_params}
                    missing = operation.missing_params(operation.params)
                    for m in missing:
                        self.pipeline[i].params[m] = self.params[m]
                    # TODO override functionality
                else:
                    print(
                        f'not enough params missing:{operation.check_missing_params({**self.params, **operation.params})}')
            else:
                raise Exception(f'not of type Operation {operation}')

    def _add_storage_pool(self):
        for operation in self.pipeline:
            operation._set_storage_pool(self.storage_pool)

    def run(self):

        if self.dfs == None:
            raise ValueError('no data added. supply data with the add_data method first!')
        if self.pipeline == []:
            raise ValueError('no pipeline added. supply a pipeline with the add_pipeline method first!')

        self._instantiate_operations_and_apply_params()
        self._add_storage_pool()

        for i in range(len(self.pipeline)):
            operation = self.pipeline[i]
            assert (isinstance(operation, Operation))

            block = deepcopy(self.xy_blocks[i])

            if isinstance(operation, PipelineOperation):
                if isinstance(operation, Back):
                    index = i-operation.steps
                    block = deepcopy(self.xy_blocks[index])

                elif isinstance(operation, BackTo):
                    index = operation.to
                    block = deepcopy(self.xy_blocks[index])

                elif isinstance(operation, BackToNamed):
                    if operation.name in self.name_to_index:
                        index = self.name_to_index[operation.name]
                        block = deepcopy(self.xy_blocks[index])
                    else:
                        raise NameError('No CheckPoint named "{}" found so far.'.format(operation.name))

                elif isinstance(operation, CheckPoint):
                    name = operation.name # is None if not named
                    self.y_checkpoints_names.append(name)

                    if isinstance(block, XYBlock):
                        self.y_checkpoints.append([block.x, block.y])
                        if name is not None:
                            self.name_to_index[name] = i
                            self.named_checkpoints[name] = [block.x, block.y]
                    else:
                        self.y_checkpoints.append([block.x, block.ya, block.yb])
                        if name is not None:
                            self.named_checkpoints[name] = [block.x, block.ya, block.yb]
                            self.name_to_index[name] = i
                else:
                    raise NotImplementedError()

            elif isinstance(operation, TransformOperation):
                block.apply_transform(operation)

            elif isinstance(operation, TransformOperationXY):
                block.apply_transform_xy(operation)

            elif isinstance(operation, SplitOperation):
                if isinstance(block, XYBlockSplit):
                    raise RuntimeError('cannot split after arleady split.')
                a = []
                b = []
                for i in range(len(self.dfs)):
                    df = self.dfs[i]
                    try:
                        result = operation.do(block.x, block.y[i,:], df)
                    except IndexError as e:
                        raise IndexError(str(e) + '\n\nSplitOperation failed: maybe only one measurement row?')
                    if result:
                        a.append(block.y[i,:])
                    else:
                        b.append(block.y[i,:])
                block = XYBlockSplit(block.x, np.stack(a), np.stack(b))

            elif isinstance(operation, CombineOperation):
                if isinstance(block, XYBlock):
                    raise RuntimeError(f'need to be split before Combine operation {operation}')

                y = operation.do(block.x, block.ya, block.yb)
                block = XYBlock(block.x, y)

            elif isinstance(operation, CollapseOperation):
                block.apply_collapse(operation)

            else:
                print('in run() operation not implemented - not ok: ' + str(operation))

            self.xy_blocks.append(block)

    def get_checkpoints(self) -> List:
        return self.y_checkpoints

    def get_checkpoint(self, index):
        return self.y_checkpoints[index]

    def get_named_checkpoint(self, name):
        return self.named_checkpoints[name]

    def get_named_checkpoints(self):
        return self.named_checkpoints

    def df_from_named(self):
        '''makes a pandas dataframe from the named checkpoints'''

        names = list(self.get_named_checkpoints().keys())
        values = list(self.get_named_checkpoints().values())

        y_values = [v[1:] for v in values]
        x_values = [v[0] for v in values]


        for x in x_values[1:]:
            if not (x_values[0] == x).all():
                raise ValueError("the x values changed. (maybe used Cut between Checkpoints?) so we can't save it one dataframe")

        y_values_new = []
        names_new = []

        for i, y in enumerate(y_values):
            if len(y) <= 1:
                y_values_new.append(y[0])
                names_new.append(names[i])
            elif len(y) > 1:
                for j, y1 in enumerate(y):
                    names_new.append(names[i] + ' ' + str(j+1))
                    y_values_new.append(y[j])

        column_names = names_new
        data = np.array(y_values_new).T

        df = pd.DataFrame(data, index=x_values[0], columns=column_names)
        return df



    def summary(self):
        raise NotImplementedError()


class MultiMeasurementProcessor:
    def __init__(self, N):
        self.singles = [SingleMeasurementProcessor() for i in range(N)]

    def add_data(self, dfs: Union[List[pd.DataFrame], List[List[pd.DataFrame]]], x_column: str, y_column: str, n_samples: int = 2000):
        for i, df in enumerate(dfs):
            self.singles[i].add_data(df, x_column, y_column, n_samples)

    def check_missing_params(self):
        for s in self.singles:
            s.check_missing_params()

    def add_params(self, params: dict, override: bool=False):
        for s in self.singles:
            s.add_params(params, override)

    def add_pipeline(self, pipeline: List[Operation]):
        for s in self.singles:
            s.add_pipeline(pipeline)

    def run(self):
        for s in self.singles:
            s.run()

    def get_checkpoints(self) -> List[List]:

        cps = [[]]*len(self.singles[0].get_checkpoints())
        for s in self.singles:
            for i, cp in enumerate(s.get_checkpoints()):
                cps[i].append(cp)
        return cps

    def summary(self):
        raise NotImplementedError()

if __name__ == '__main__':
    p = SingleMeasurementProcessor()
    p.add_pipeline()
    p.add_params()
    p.add_data()
    p.run()

    print(p.summary())