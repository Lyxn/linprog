import numpy as np

from decomposition import simplex_dantzig_wolfe
from linprog import linprog
from utils import to_array, to_array_list

DEBUG = True


def test_dw0_revised():
    c = [-4, -1, -3, -2]
    b = [6, 4, 5, 1, 2, 6]
    A = [[2, 2, 1, 2], [0, 1, 2, 3], [2, 1, 0, 0], [0, 1, 0, 0], [0, 0, -1, 2], [0, 0, 1, 2]]
    c = to_array(c)
    A = to_array(A)
    b = to_array(b)
    opt_val = -14
    print("\nTest Dantzig Wolfe 0")
    ret = linprog(c, A_ub=A, b_ub=b, debug=DEBUG)
    assert type(ret) != int and ret.z_opt == opt_val


def test_dw0_dantzig():
    c0 = [[-4, -1, 0, 0], [-3, -2, 0, 0]]
    b0 = [6, 4]
    b = [[5, 1], [2, 6]]
    L1 = [[2, 2, 0, 0], [0, 1, 0, 0]]
    L2 = [[1, 2, 0, 0], [2, 3, 0, 0]]
    A1 = [[2, 1, 1, 0], [0, 1, 0, 1]]
    A2 = [[-1, 2, 1, 0], [1, 2, 0, 1]]
    basis = [[2, 3], [2, 3]]
    opt_val = -14
    c0 = to_array_list(c0)
    b0 = to_array(b0)
    b = to_array_list(b)
    L = [L1, L2]
    A = [A1, A2]
    L = to_array_list(L)
    A = to_array_list(A)
    print("\nTest Dantzig Wolfe 0")
    ret = simplex_dantzig_wolfe(c0, L, b0, A, b, basis=basis, debug=DEBUG)
    assert type(ret) != int and ret.z_opt == opt_val


def test_dw1_dantzig():
    c0 = [[-1, -2, 0, 0], [-4, -3, 0, 0]]
    b0 = [4, 3]
    b = [[4, 2], [2, 5]]
    L1 = [[1, 1, 0, 0], [0, 1, 0, 0]]
    L2 = [[2, 0, 0, 0], [1, 1, 0, 0]]
    A1 = [[2, 1, 1, 0], [1, 1, 0, 1]]
    A2 = [[1, 1, 1, 0], [3, 2, 0, 1]]
    basis = [[2, 3], [2, 3]]
    opt_val = -10
    c0 = to_array_list(c0)
    b0 = to_array(b0)
    b = to_array_list(b)
    L = [L1, L2]
    A = [A1, A2]
    L = to_array_list(L)
    A = to_array_list(A)
    print("\nTest Dantzig Wolfe 1")
    ret = simplex_dantzig_wolfe(c0, L, b0, A, b, basis=basis, debug=DEBUG)
    assert type(ret) != int and ret.z_opt == opt_val


if __name__ == '__main__':
    # test_dw0_revised()
    test_dw0_dantzig()
    test_dw1_dantzig()
