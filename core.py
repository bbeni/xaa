from copy import deepcopy
import sys
import pandas as pd
import numpy as np
from typing import Union, List, Iterator

from operations import Operation, PipelineOperation, TransformOperation, SplitOperation, CombineOperation, CollapseOperation
from operations import Back, BackTo, CheckPoint

from helpers import common_bounds, interpolate, StoragePool


class XYBlock:
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
            raise NotImplementedError('level ' + self.level())

    def apply_collapse(self, collapse: CollapseOperation):
        self.y = collapse.do(self.x, self.y)

class XYBlockSplit:
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
    def __init__(self):
        self.processed = False
        self.dfs = None

        self.xy_blocks = []

        self.override_params = False
        self.params = {}
        self.pipeline = []

        self.y_chekpoints = []
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
            print('No Parameter is missing. Nice!')
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

                elif isinstance(operation, CheckPoint):

                    if isinstance(block, XYBlock):
                        self.y_chekpoints.append(block.y)
                    else:
                        self.y_chekpoints.append([block.ya, block.yb])
                else:
                    raise NotImplementedError()

            elif isinstance(operation, TransformOperation):
                block.apply_transform(operation)

            elif isinstance(operation, SplitOperation):
                if isinstance(block, XYBlockSplit):
                    raise RuntimeError('cannot split after arleady split.')
                a = []
                b = []
                for i in range(len(self.dfs)):
                    df = self.dfs[i]
                    result = operation.do(block.x, block.y[i,:], df)
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
                print('not ok')

            self.xy_blocks.append(block)

    def get_checkpoints(self) -> List:
        return self.y_chekpoints

    def get_checkpoint(self, index):
        return self.y_chekpoints[index]

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