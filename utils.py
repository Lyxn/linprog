# encode: utf8
from __future__ import print_function

import numpy as np


def to_array(x):
    return np.array(x, dtype=float)


def to_array_list(lst):
    return [to_array(x) for x in lst]


def get_unit_vector(dim, idx):
    unit = np.zeros(dim)
    unit[idx] = 1
    return unit


def is_pos(x, eps=1e-10):
    return x > eps


def is_neg(x, eps=1e-10):
    return x < -eps


def is_zero(x, eps=1e-10):
    return abs(x) <= eps


def is_pos_zero(x, eps=1e-10):
    return is_pos(x, eps) or is_zero(x, eps)


def is_neg_zero(x, eps=1e-10):
    return is_neg(x, eps) or is_zero(x, eps)


def is_pos_all(arr, eps=1e-10):
    return all(is_pos(x) for x in arr)


def is_neg_all(arr, eps=1e-10):
    return all(is_neg(x) for x in arr)


def is_integer(x, eps=1e-10):
    return is_zero(x - round(x))


def is_integer_list(arr, eps=1e-10):
    return all(is_integer(i, eps) for i in arr)


def floor_residue(x):
    return x - np.floor(x)


def take_index(arr, idxs):
    return [arr[i] for i in idxs]
