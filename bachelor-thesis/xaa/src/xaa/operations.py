import numpy as np
import pandas as pd
import scipy.integrate as integrate
from lmfit import Model

from .helpers import closest_idx, peak_x


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

    def do(self, x, y):
        raise NotImplementedError()

    def post(self, x, y):
        return y

    # ugly
    def get_global(self, key):
        return self.storage_pool.get(key)

    def set_global(self, key, value):
        self.storage_pool.save(key, value)

    def _set_storage_pool(self, storage_pool):
        self.storage_pool = storage_pool


class PipelineOperation(Operation):
    pass

class TransformOperation(Operation):
    def do(self, x, y):
        raise NotImplementedError()
        return y
    def clear_temp(self):
        pass

class TransformOperationXY(Operation):
    def do(self, x, y):
        raise NotImplementedError()
        return x, y
    def clear_temp(self):
        pass

class SplitOperation(Operation):
    def do(self, x, y, df: pd.DataFrame) -> bool:
        raise NotImplementedError()

class CombineOperation(Operation):
    def do(self, x, ya, yb):
        raise NotImplementedError()

class CollapseOperation(Operation):
    def do(self, x, y):
        raise NotImplementedError()


### Implementations PipelineOperations

class Back(PipelineOperation):
    def __init__(self, steps=1):
        super().__init__()
        self.steps = steps

class BackTo(PipelineOperation):
    def __init__(self, to=0):
        super().__init__()
        self.to = to

class BackToNamed(PipelineOperation):
    def __init__(self, name:str):
        super().__init__()
        self.name = name

class Input(PipelineOperation):
    def __init__(self, name):
        super(Input, self).__init__()
        self.name = name

class CheckPoint(PipelineOperation):
    '''creates a checkpoint that stores the state.
       can be reverted to with BackToNamed('name')
    '''
    def __init__(self, name:str):
        super(CheckPoint, self).__init__()
        self.name = name


### Implementations Data changing operations

class Average(CollapseOperation):
    def do(self, x, y):
        return np.average(y, axis=0)

class Flip(TransformOperation):
    def do(self, x, y):
        return -y

class ApplyFunctionToY(TransformOperation):
    expected_params = ['function']
    def __init__(self, function):
        super().__init__(function=function)
    def do(self, x, y):
        return self.get_param('function')(y)

class Add(TransformOperation):
    def __init__(self, c):
        super().__init__()
        self.c = c

    def do(self, x, y):
        return y + self.c

class Multiply(TransformOperation):
    def __init__(self, c):
        super(Multiply, self).__init__()
        self.c = c

    def do(self, x, y):
        return y*self.c

class Normalize(TransformOperation):
    def __init__(self, save=None, to=None, use_min=False):
        super().__init__()
        if save and to:
            raise ValueError('save and to can not be used at the same time')
        self.save = save
        self.to = to
        self.use_min = use_min

    def do(self, x, y):
        f = np.min if self.use_min else np.max
        if self.save:
            m = f(y)
            self.set_global(self.save, m)
        elif self.to:
            m = self.get_global(self.to)
        else:
            m = f(y)
        y /= m
        return y

class NormalizePeak(TransformOperation):
    expected_params = ['normalizepeak_range']
    def __init__(self, save=None):
        super().__init__()
        self.save = save

    def do(self, x, y):
        peak_range = self.get_param('normalizepeak_range')
        ia, ib = closest_idx(x, peak_range[0]), closest_idx(x, peak_range[1])
        assert (ia < ib)
        m = np.max(y[ia:ib])
        if self.save:
            self.set_global(self.save, m)
        y /= m
        return y

class PreEdgeScaleToFirst(TransformOperation):
    expected_params = ['pre_edge_range']

    def clear_temp(self):
        self.is_first = True

    def do(self, x, y):
        pre_edge_range = self.get_param('pre_edge_range')
        ia, ib = closest_idx(x, pre_edge_range[0]), closest_idx(x, pre_edge_range[1])
        assert(ia < ib)    
        mean = np.mean(y[ia:ib])

        if self.is_first:
            self.is_first = False
            self.first_mean = mean
            return y
        else:
            return self.first_mean / mean * y

class PreAndPostEdgeScaleToFirst(TransformOperation):
    expected_params = ['pre_edge_range', 'post_edge_range', 'pre_weight']

    def clear_temp(self):
        self.is_first = True

    def do(self, x, y):
        pre_edge_range = self.get_param('pre_edge_range')
        post_edge_range = self.get_param('post_edge_range')
        weight = self.get_param('pre_weight')

        ia, ib = closest_idx(x, pre_edge_range[0]), closest_idx(x, pre_edge_range[1])
        assert(ia < ib)    
        mean_pre = np.mean(y[ia:ib])
        ia, ib = closest_idx(x, post_edge_range[0]), closest_idx(x, post_edge_range[1])
        assert(ia < ib)    
        mean_post = np.mean(y[ia:ib])
        mean = weight*mean_pre + (1-weight)*mean_post

        if self.is_first:
            self.is_first = False
            self.first_mean = mean
            return y
        else:
            return self.first_mean / mean * y


class BringToZero(TransformOperation):
    expected_params = ['to_zero_range']
    def do(self, x, y):
        to_zero_range = self.get_param('to_zero_range')
        ia, ib = closest_idx(x, to_zero_range[0]), closest_idx(x, to_zero_range[1])
        m = np.mean(y[ia:ib])
        return y - m




def fermi_step(x, y, peak_1, peak_2, post, delta, a):
    def mu_step(x, E_l3, E_l2, step_e, delta, h, a):
        b = 1-a
        return h*(1- a*(1/(1+np.exp((x-E_l3-step_e)/delta))) - b*(1/(1+np.exp((x-E_l2-step_e)/delta))))

    post_range = closest_idx(x, post[0]), closest_idx(x, post[1])

    try:
        idx_l3 = peak_x(x, y, peak_2)
    except ValueError as e:
        print(e)
        raise ValueError('In FermiBg: pipeline params wrong \n\n param peak_2 did not find a y value in range of x={}'.format(peak_2))
    try:
        idx_l2 = peak_x(x, y, peak_1)
    except ValueError as e:
        print(e)
        raise ValueError('In FermiBg: pipline params wrong \n\n param peak_1 did not find a y value in range of x={}'.format(peak_1))

    h = np.mean(y[post_range[0]:post_range[1]])
    f = lambda x: mu_step(x, idx_l3, idx_l2, 0, delta, h, a)
    return f(x)

class FermiBG(TransformOperation):
    expected_params = ['peak_1', 'peak_2', 'post', 'delta', 'a']

    def do(self, x, y):
        fermi_bg = fermi_step(x, y, self.get_param('peak_1'), self.get_param('peak_2'),
                          self.get_param('post'),
                          self.get_param('delta'), self.get_param('a'))

        return y - fermi_bg


def fermi_step_fitted(x, y, peak_1, peak_2, mid, post, delta):
    def mu_step(x, E_l3, E_l2, step_e, delta, h, a):
        b = 1-a
        return h*(1- a*(1/(1+np.exp((x-E_l3-step_e)/delta))) - b*(1/(1+np.exp((x-E_l2-step_e)/delta))))

    post_range = closest_idx(x, post[0]), closest_idx(x, post[1])
    mid_range = closest_idx(x, mid[0]), closest_idx(x, mid[1])

    idx_l3 = peak_x(x, y, peak_2)
    idx_l2 = peak_x(x, y, peak_1)

    h1 = np.mean(y[mid_range[0]:mid_range[1]])
    h2 = np.mean(y[post_range[0]:post_range[1]])

    a = 1 - h1/h2

    f = lambda x: mu_step(x, idx_l3, idx_l2, 0, delta, h2, a)
    return f(x)

class FermiBGfittedA(TransformOperation):
    '''like a fermi double step but with a fitted'''
    expected_params = ['peak_1', 'peak_2', 'mid', 'post', 'delta']
    def do(self, x, y):
        fermi_bg = fermi_step_fitted(x, y, self.get_param('peak_1'), self.get_param('peak_2'),
                          self.get_param('mid'), self.get_param('post'),
                          self.get_param('delta'))

        return y - fermi_bg


def fermi_step_adaptive(x, y, peak_1, peak_2, mid, post, delta):
    """
    plan:
    1. fit 2 cosh to mid and post -> 6 parameters
    2. calculate step*cosh1 + step*cosh2
    """

    def cosh_like(x, a, b, h):
        return (1 - np.cosh((x - b) / 10)) / a + h

    def fermi_wall(x, a, b, delta):
        return 1 / (1 + np.exp((-x + a) / delta)) + 1 / (1 + np.exp((x - b) / delta)) - 1

    post_range = closest_idx(x, post[0]), closest_idx(x, post[1])
    mid_range = closest_idx(x, mid[0]), closest_idx(x, mid[1])

    idx_l3 = peak_x(x, y, peak_1)
    idx_l2 = peak_x(x, y, peak_2)

    model1 = Model(cosh_like)
    model2 = Model(cosh_like)
    model1.set_param_hint('a', value=-1, min=-100, max=-0.01)
    model1.set_param_hint('b', value=750, min=700, max=900)
    model2.set_param_hint('a', value=-0.1, min=-100, max=100)
    model2.set_param_hint('b', value=850, min=750, max=950)

    p1 = model1.make_params(h=0.5)
    p2 = model2.make_params(h=1)

    f1 = model1.fit(y[mid_range[0]: mid_range[1]], p1, x=x[mid_range[0]:mid_range[1]])
    f2 = model2.fit(y[post_range[0]: post_range[1]], p2, x=x[post_range[0]:post_range[1]])

    wall1 = fermi_wall(x, a=idx_l3, b=idx_l2, delta=delta)
    wall2 = fermi_wall(x, a=idx_l2, b=10000, delta=delta)

    #print(idx_l2, idx_l3)
    #print(f1.fit_report())
    #print(f2.fit_report())

    fitted = f1.eval(x=x) * wall1 + f2.eval(x=x) * wall2

    return fitted


class FermiWallAdaptive(TransformOperation):
    expected_params = ['peak_1', 'peak_2', 'mid', 'post', 'delta']

    def do(self, x, y):
        fermi_bg = fermi_step_adaptive(x, y, self.get_param('peak_1'), self.get_param('peak_2'),
                          self.get_param('mid'), self.get_param('post'),
                          self.get_param('delta'))

        return y - fermi_bg


def fermi_step_adaptive2(x, y, peak_1, peak_2, pre, mid, post, delta):
    """
    plan:
    1. fit 2 cosh to mid and post -> 6 parameters
    2. calculate step*cosh1 + step*cosh2
    """

    def cosh_like(x, a, b, h):
        return (1 - np.cosh((x - b) / 10)) / a + h

    def fermi_wall(x, a, b, delta):
        return 1 / (1 + np.exp((-x + a) / delta)) + 1 / (1 + np.exp((x - b) / delta)) - 1

    post_range = closest_idx(x, post[0]), closest_idx(x, post[1])
    mid_range = closest_idx(x, mid[0]), closest_idx(x, mid[1])
    pre_range = closest_idx(x, pre[0]), closest_idx(x, pre[1])

    idx_l3 = peak_x(x, y, peak_1)
    idx_l2 = peak_x(x, y, peak_2)

    model1 = Model(cosh_like)
    model2 = Model(cosh_like)
    model3 = Model(cosh_like)

    model1.set_param_hint('a', value=-1, min=-100, max=-0.01)
    model1.set_param_hint('b', value=750, min=700, max=900)
    model2.set_param_hint('a', value=-0.1, min=-100, max=100)
    model2.set_param_hint('b', value=850, min=750, max=950)
    model3.set_param_hint('a', value=-0.1, min=-100, max=100)
    model3.set_param_hint('b', value=850, min=750, max=950)

    p1 = model1.make_params(h=0.5)
    p2 = model2.make_params(h=1)
    p3 = model1.make_params(h=0)

    f1 = model1.fit(y[mid_range[0]: mid_range[1]], p1, x=x[mid_range[0]:mid_range[1]])
    f2 = model2.fit(y[post_range[0]: post_range[1]], p2, x=x[post_range[0]:post_range[1]])
    f3 = model2.fit(y[pre_range[0]: pre_range[1]], p3, x=x[pre_range[0]:pre_range[1]])


    wall1 = fermi_wall(x, a=idx_l3, b=idx_l2, delta=delta)
    wall2 = fermi_wall(x, a=idx_l2, b=10000, delta=delta)
    wall3 = fermi_wall(x, a=0, b=idx_l3, delta=delta)

    #print(idx_l2, idx_l3)
    print(f1.fit_report())
    print(f2.fit_report())
    print(f3.fit_report())

    fitted = f1.eval(x=x) * wall1 + f2.eval(x=x) * wall2 + f3.eval(x=x) * wall3

    return fitted



class FermiWallAdaptive2(TransformOperation):
    expected_params = ['peak_1', 'peak_2', 'pre', 'mid', 'post', 'delta']

    def do(self, x, y):
        fermi_bg = fermi_step_adaptive2(x, y, self.get_param('peak_1'), self.get_param('peak_2'),
                          self.get_param('pre'), self.get_param('mid'), self.get_param('post'),
                          self.get_param('delta'))

        return y - fermi_bg


def gauss_bg_fit(x, y, fit_range):
    ia, ib = closest_idx(x, fit_range[0]), closest_idx(x, fit_range[1])
    print(ia, ib)
    peak = peak_x(x, y, fit_range)
    print(peak)

    def gauss(x, a, b, c):
        return a * np.exp(-(x - b) ** 2 / c)

    def pseudo_voigt(x, a, eta, b, c):
        return a*(eta*np.exp(-np.log(2)*(x-b)**2/c) + (1-eta)* 1/(1+(x-b)**2/c))

    m = Model(gauss)
    m.set_param_hint('b', value=peak, min=peak-2, max=peak+2)
    p = m.make_params(b=peak, a=1, c=0.1)

    m = Model(pseudo_voigt)
    m.set_param_hint('b', value=peak, min=peak-2, max=peak+2)
    m.set_param_hint('eta', value=0.5, min=0, max=1)
    p = m.make_params(b=peak, a=1, c=0.1)

    fitted = m.fit(y[ia:ib], p, x=x[ia:ib])
    print(fitted.fit_report())

    return fitted.eval(x=x)


class GaussianFitRemove(TransformOperation):
    expected_params = ['gauss_fit_range']

    def do(self, x, y):
        bg = gauss_bg_fit(x, y, self.get_param('gauss_fit_range'))
        return y-bg


def arctan_adaptive_fit(x, y, pre, post):

    post_range = closest_idx(x, post[0]), closest_idx(x, post[1])
    pre_range = closest_idx(x, pre[0]), closest_idx(x, pre[1])

    def arctan(x, a, b, c, d):
        return a*np.arctan((x-b)*c)+d

    model = Model(arctan)
    p = model.make_params(a=0.5, b=851, c=1, d=3)


    xa = x[pre_range[0]:pre_range[1]]
    xb = x[post_range[0]:post_range[1]]

    ya = y[pre_range[0]:pre_range[1]]
    yb = y[post_range[0]:post_range[1]]

    x_hat = np.concatenate((xa, xb))
    y_hat = np.concatenate((ya, yb))


    fitted = model.fit(y_hat, p, x=x_hat)
    #print(fitted.fit_report())

    return fitted.eval(x=x)

class ArctanAdaptiveRemove(TransformOperation):
    expected_params = ['atan_adapt_pre', 'atan_adapt_post']
    def do(self, x, y):
        pre = self.get_param('atan_adapt_pre')
        post = self.get_param('atan_adapt_post')

        # heurisitic setting to arctan
        post_average = post[0] / 2 + post[1] / 2
        pre_average = pre[0] / 2 + pre[1] / 2
        a, b = closest_idx(x, pre_average), closest_idx(x, post_average)

        atan_y = arctan_adaptive_fit(x, y, pre, post)
        y[a:b] = atan_y[a:b]
        return y



class Cut(TransformOperationXY):
    expected_params = ['cut_range']
    def do(self, x, y):
        x_min, x_max = self.get_param('cut_range')
        ia, ib = closest_idx(x, x_min), closest_idx(x, x_max)
        if (ib <= ia):
            ib = ia+2

        return x[ia:ib], y[ia:ib]


class LineBGRemove(TransformOperation):
    expected_params = ['line_range']
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

class AvgLineBGRemove(TransformOperation):
    expected_params = ['line_range']
    def clear_temp(self):
        self.sum = None
        self.n = None 

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

        if self.sum is None:
            self.sum = result
            self.n = 1
        else:
            self.sum += result
            self.n += 1
        
        # just do nothing and in post remove the same line for every measurement
        return y

    def post(self, x, y):
        assert(self.sum is not None)
        avg = self.sum/self.n
        return y - avg



class Integrate(TransformOperation):
    def do(self, x, y):
        integral = integrate.cumulative_trapezoid(y, x, initial=0)
        return integral

class CombineDifference(CombineOperation):
    def do(self, x, ya, yb):
        return yb-ya

class CombineAverage(CombineOperation):
    def do(self, x, ya, yb):
        return (ya+yb)/2

class SplitBy(SplitOperation):
    expected_params = ['binary_filter']
    def __init__(self, binary_filter):
        super().__init__(binary_filter=binary_filter)
    def do(self, x, y, df):
        filter = self.get_param('binary_filter')
        return filter(df)


