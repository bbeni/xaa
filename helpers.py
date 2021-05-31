import numpy as np
from scipy.interpolate import interp1d

def interpolate(x, y, new_x_space):
    x, index = np.unique(x, return_index=True)
    y = y[index]
    f = interp1d(x, y)
    return f(new_x_space)

def common_bounds(x):
    return max(map(np.min, x)), min(map(np.max, x))

def closest_idx(space, value):
    return np.abs(space-value).argmin()

def peak_x(x, y, x_range):
    return x[peak_index(x, y, x_range)]

def peak_index(x, y, x_range):
    i = closest_idx(x, x_range[0])
    j = closest_idx(x, x_range[1])
    idx = np.argmax(y[i:j]) + i
    return idx


class StoragePool:
    def __init__(self):
        self.storage = {}

    def save(self, key, value):
        if key in self.storage:
            raise KeyError(f'{key} already in global storage!')
        self.storage[key] = value

    def get(self, key):
        return self.storage[key]