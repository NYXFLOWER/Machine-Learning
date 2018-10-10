import numpy as np


def sum_squares_error(Y, X, M, C):
    diff = Y - M*X - C
    return np.sum(np.power(diff, 2))


def offset_update(Y, X, M):
    return np.average(Y - M*X)


def gradient_update(Y, X, C):
    numerate = np.sum((Y - C)*X)
    denominator = np.sum(np.power(X, 2))
    return numerate/denominator
