import numpy as np
from math import isclose, sqrt

def test_interpolate_basic():
    """
    tests basic interpolation
    """
    xs = np.array([0, 1, 2])
    ys = np.array([0, 1, 2])
    xt = np.array([0.5, 1.5])
    yt = np.interp(xt, xs, ys)
    assert isclose(yt[0], 0.5)
    assert isclose(yt[1], 1.5)

def test_threshold_calc():
    """
    tests threshold calculation
    """
    train_max = 1.0
    thr = train_max * sqrt(2)
    assert isclose(thr, sqrt(2))