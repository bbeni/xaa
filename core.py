from copy import deepcopy
import pandas as pd
import numpy as np
from typing import Union

from operations import Operation, PipelineOperation, Transform, Split, Combine, Collapse
from operations import Back, BackTo, CheckPoint

from helpers import common_bounds, interpolate, StoragePool


class XYBlock:
    def __init__(self, x, y, name=''):
        self.x = x
        self.y = y
        self.name = name
        self.state = 'normal'

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

    def add_data(self, dfs: Union[pd.DataFrame, list[pd.DataFrame]], x_column: str, y_column: str, n_samples: int = 2000):
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

    def missing_params(self):
        if self.pipeline == []:
            raise ValueError('no pipeline added. supply a pipeline with the add_pipeline method first!')

        for i, operation in enumerate(self.pipeline):
            if isinstance(operation, Operation):
                missing = operation.missing_params({**self.params, **operation.params})
                if missing:
                    print(f'Missing Parameter(s) in {operation} pos[{i}]')
                    print(missing)
            else:
                missing = operation.missing_params(self.params)
                if missing:
                    print(f'Missing Parameter(s) in {operation} pos[{i}]')
                    print('Operation not instanciated yet. looked only in global parameters...')
                    print(missing)

    def add_params(self, params: dict, override: bool=False):
        self.override_params = override
        self.params = params

    def add_pipeline(self, pipeline: list[Operation]):
        for operation in pipeline:
            if isinstance(operation, type):
                if not issubclass(operation, Operation):
                    raise ValueError(f'{operation} is a class but not a subclass of Operation')
            elif isinstance(operation, Operation):
                pass
            else:
                raise ValueError(f'{operation} is an object is not an instance of Operation')

            self.pipeline.append(operation)


    def _instanciate_operations_and_apply_params(self):
        for i in range(len(self.pipeline)):
            operation = self.pipeline[i]
            if isinstance(operation, type):
                # find global params and instantiate
                p = {k:v for k,v in self.params if k in operation.expected_params}
                self.pipeline[i] = operation(p)
            elif isinstance(operation, Operation):
                if operation.check_params(operation.params):
                    # TODO override functionality
                    continue
                elif operation.check_params({**self.params, **operation.params}):
                    p = {k: v for k, v in self.params if k in operation.expected_params}
                    missing = operation.missing_params(operation.params)
                    for m in missing:
                        self.pipeline[i].params[m] = self.params[m]
                    # TODO override functionality
                else:
                    print(f'not enough params missing:{operation.missing_params({**self.params, **operation.params})}')
            else:
                raise Exception(f'not of type Operation {operation}')


    def run(self):
        if self.dfs == None:
            raise ValueError('no data added. supply data with the add_data method first!')
        if self.pipeline == []:
            raise ValueError('no pipeline added. supply a pipeline with the add_pipeline method first!')

        self._instanciate_operations_and_apply_params()

        states = ['normal', 'split']

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

                    if block.state == 'normal':
                        self.y_chekpoints.append(block.y)
                    else:
                        self.y_chekpoints.append(block.ya, block.yb)
                else:
                    raise NotImplementedError()

            elif isinstance(operation, Transform):
                if block.state == 'normal':
                    block.y = operation(block.x, block.y)
                elif block.state == 'split':
                    block.ya = operation(block.x, block.ya)
                    block.yb = operation(block.x, block.yb)

            elif isinstance(operation, Split):
                if block.state == 'split':
                    raise RuntimeError('cannot split after arleady split.')
                a = []
                b = []
                for i in range(len(self.dfs)):
                    df = self.dfs[i]
                    result = operation(block.x, block.y[i,:], df)
                    if result:
                        a.append(block.y[i,:])
                    else:
                        b.append(block.y[i,:])
                block.ya = np.stack(a)
                block.yb = np.stack(b)
                del block.y
                block.state = 'split'

            elif isinstance(operation, Combine):
                if block.state == 'normal':
                    raise RuntimeError(f'need to be split before Combine operation {operation}')

                y = operation(block.x, block.ya, block.yb)
                block.y = y
                del block.ya, block.yb
                block.state = 'normal'

            elif isinstance(operation, Collapse):
                raise NotImplementedError()

            else:
                print('not ok')

            self.xy_blocks.append(block)

    def get_checkpoints(self):
        return self.y_chekpoints

    def summary(self):
        raise NotImplementedError()


if __name__ == '__main__':
    p = SingleMeasurementProcessor()
    p.add_pipeline()
    p.add_params()
    p.add_data()
    p.run()

    print(p.summary())