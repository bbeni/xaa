import numpy as np
import pandas as pd
from helpers import closest_idx, peak_x
import scipy.integrate as integrate
from lmfit import Model


class Operation:
    expected_params = []

    def __init__(self, **params):
        self.params = params

    def get_param(self, name):
        return self.params[name]

    @classmethod
    def check_params(cls, params):
        for param_name in cls.expected_params:
            if param_name not in params.keys():
                return False
        return True

    @classmethod
    def missing_params(cls, params):
        missing = []
        for param_name in cls.expected_params:
            if param_name not in params.keys():
                missing.append(param_name)
        return missing

    def do(self):
        raise NotImplementedError()

    # ugly
    def get_global(self, key):
        return self.storage_pool.get(key)

    def set_global(self, key, value):
        self.storage_pool.save(key, value)

    def _set_storage_pool(self, storage_pool):
        self.storage_pool = storage_pool


class PipelineOperation(Operation):
    pass

class Transform(Operation):
    def do(self, x, y):
        return y

class Split(Operation):
    def do(self, x, y, df: pd.DataFrame) -> bool:
        raise NotImplementedError()

class Combine(Operation):
    def do(self, x, ya, yb):
        raise NotImplementedError()

class Collapse(Operation):
    def do(self, x, y):
        raise NotImplementedError()


class Back(PipelineOperation):
    def __init__(self, steps=1):
        self.steps = steps

class BackTo(PipelineOperation):
    def __init__(self, to=0):
        self.to = to

class CheckPoint(PipelineOperation):
    pass

class Average(Collapse):
    def do(self, x, y):
        return x, np.average(y, axis=0)

class Flip(Transform):
    def do(self, x, y):
        return x, -y

class Normalize(Transform):
    def __init__(self, save=None, to=None):
        if save and to:
            raise ValueError('save and to can not be used at the same time')
        self.save = save
        self.to = to

    def do(self, x, y):
        if self.save:
            m = np.max(y)
            self.set_global(self.save, m)
        elif self.to:
            m = self.get_global(self.to)
        y /= m
        return y


def fermi_step(x, y, peak_1, peak_2, post, delta, a):
    def mu_step(x, E_l3, E_l2, step_e, delta, h, a):
        b = 1-a
        return h*(1- a*(1/(1+np.exp((x-E_l3-step_e)/delta))) - b*(1/(1+np.exp((x-E_l2-step_e)/delta))))

    post_range = closest_idx(x, post[0]), closest_idx(x, post[0])
    idx_l3 = peak_x(x, y, peak_2)
    idx_l2 = peak_x(x, y, peak_1)

    h = np.mean(y[post_range[0]:post_range[1]])
    f = lambda x: mu_step(x, idx_l3, idx_l2, 0, delta, h, a)
    return f(x)

class FermiBG(Transform):
    expected_params = ['peak_1', 'peak_2', 'post', 'delta', 'a']
    def do(self, x, y):
        fermi_bg = fermi_step(x, y, self.get_param('peak_1'), self.get_param('peak_2'),
                          self.get_param('post'),
                          self.get_param('delta'), self.get_param('a'))

        return y - fermi_bg


class LineBG(Transform):
    expected = ['line_range']
    def do(self, x, y):
        def line(x, slope, intercept):
            return slope*x + intercept
        model = Model(line)
        p = model.make_params(intercept=-1, slope=0.001)
        ia, ib = closest_idx(x, self.get_param('line_range')[0]), closest_idx(x, self.get_param('line_range')[1])
        if (ib <= ia):
            ib = ia+2
        fitted = model.fit(y[ia:ib], p, x=x[ia:ib])
        result = fitted.eval(x=x)
        return y - result

class Integrate(Transform):
    def do(self, x, y):
        integral = integrate.cumulative_trapezoid(y, x, initial=0)
        return integral

class Difference(Combine):
    def do(self, x, ya, yb):
        return yb-ya

class SplitBy(Split):
    expected_params = ['binary_filter']
    def do(self, x, y, df):
        filter = self.get_param('binary_filter')
        return filter(df)


