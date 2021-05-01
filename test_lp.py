# encode=utf8
from __future__ import print_function

import numpy as np

from linprog import linprog_primal
from simplex import simplex_dual
from simplex import simplex_revised
from utils import to_array

DEBUG = True


def test_revised():
    """
    min c * x
    s.t A * x <= b,
        x >= 0.
    """
    c = [-3, -1, -3]
    b = [2, 5, 6]
    A = [[2, 1, 1], [1, 2, 3], [2, 2, 1]]
    basis = [3, 4, 5]
    num_slack = 3
    opt_val = -5.4
    c = to_array(c)
    c = np.concatenate((c, np.zeros(num_slack)))
    A = np.concatenate((A, np.eye(num_slack)), axis=1)
    print("\nTest Revised Simplex")
    ret = simplex_revised(c, A, b, basis, debug=DEBUG)
    print(np.dot(ret.x_opt, c))
    assert type(ret) != int and ret.z_opt == opt_val
    print("\nTest Two Phrase Method")
    ret = linprog_primal(c, A, b, debug=DEBUG)
    assert type(ret) != int and ret.z_opt == opt_val


def test_dual0():
    """
    min c * x
    s.t A * x >= b
        x >= 0.
    """
    c = [3, 4, 5]
    b = [5, 6]
    A = [[1, 2, 3], [2, 2, 1]]
    basis = [3, 4]
    opt_val = 11
    b = to_array(b)
    num_slack = 2
    c = np.concatenate((c, np.zeros(num_slack)))
    A = np.concatenate((A, -np.eye(num_slack)), axis=1)
    print("\nTest Dual Simplex 0")
    ret = simplex_dual(c, -A, -b, basis, debug=DEBUG)
    assert type(ret) != int and ret.z_opt == opt_val


def test_dual1():
    """
    min c * x
    s.t A * x <= b
        x >= 0.
    """
    c = [12, 8, 16, 12]
    b = [-2, -3]
    A = [[-2, -1, -4, 0], [-2, -2, 0, -4]]
    basis = [4, 5]
    b = to_array(b)
    num_slack = 2
    opt_val = 14
    c = np.concatenate((c, np.zeros(num_slack)))
    A = np.concatenate((A, np.eye(num_slack)), axis=1)
    print("\nTest Dual Simplex 1")
    ret = simplex_dual(c, A, b, basis, debug=DEBUG, max_iter=10)
    assert type(ret) != int and ret.z_opt == opt_val


if __name__ == "__main__":
    test_revised()
    test_dual0()
    test_dual1()
