# encode: utf8
import sys
import numpy as np

def get_unit_vector(dim, idx):
    unit = np.zeros(dim)
    unit[idx] = 1
    return unit


def is_pos(x, eps=1e-16):
    return x > eps


def is_neg(x, eps=1e-16):
    return x < -eps


def is_zero(x, eps=1e-16):
    return abs(x) <= eps


def is_integer(x, eps=1e-16):
    return is_zero(x - round(x))


def floor_residue(x):
    return x - np.floor(x)
