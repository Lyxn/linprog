# encode=utf8
from __future__ import print_function

import numpy as np

from decomposition import simplex_dantzig_wolfe
from linprog import find_null_variable
from linprog import linprog
from linprog import linprog_primal
from linprog import reduce_equation
from simplex import simplex_dual
from simplex import simplex_revised


def test_revised():
    c = [-3, -1, -3]
    b = [2, 5, 6]
    A = [[2, 1, 1], [1, 2, 3], [2, 2, 1]]
    basis = [3, 4, 5]
    c = np.array(c)
    A = np.array(A)
    b = np.array(b)
    print("\nTest linprog")
    print(linprog(c, A_ub=A, b_ub=b))
    basis = np.array(basis)
    num_slack = 3
    c = np.concatenate((c, np.zeros(num_slack)))
    A = np.concatenate((A, np.eye(num_slack)), axis=1)
    print("\nTest Revised Simplex")
    print(simplex_revised(c, A, b, basis, debug=True))
    print("\nTest Two Phrase Method")
    print(linprog_primal(c, A, b, debug=True))


def test_dual0():
    c = [3, 4, 5]
    b = [5, 6]
    A = [[1, 2, 3], [2, 2, 1]]
    basis = [3, 4]
    b = np.array(b)
    basis = np.array(basis)
    num_slack = 2
    c = np.concatenate((c, np.zeros(num_slack)))
    A = np.concatenate((A, -np.eye(num_slack)), axis=1)
    print("\nTest Dual Simplex")
    print(simplex_dual(c, -A, -b, basis, debug=True))


def test_dual1():
    c = [12, 8, 16, 12]
    b = [-2, -3]
    A = [[-2, -1, -4, 0], [-2, -2, 0, -4]]
    basis = [4, 5]
    b = np.array(b)
    basis = np.array(basis)
    num_slack = 2
    c = np.concatenate((c, np.zeros(num_slack)))
    A = np.concatenate((A, np.eye(num_slack)), axis=1)
    print("\nTest Dual Simplex")
    print(simplex_dual(c, A, b, basis, debug=True, max_iter=10))


def test_dwp0_revised():
    c = [-4, -1, -3, -2]
    b = [6, 4, 5, 1, 2, 6]
    A = [[2, 2, 1, 2], [0, 1, 2, 3], [2, 1, 0, 0], [0, 1, 0, 0], [0, 0, -1, 2], [0, 0, 1, 2]]
    c = np.array(c)
    A = np.array(A)
    b = np.array(b)
    print("\nTest Dantzig Wolfe 0")
    print(linprog(c, A_ub=A, b_ub=b, debug=True))


def test_dwp0_dantzig():
    c0 = [[-4, -1, 0, 0], [-3, -2, 0, 0]]
    b0 = [6, 4]
    b = [[5, 1], [2, 6]]
    L1 = [[2, 2, 0, 0], [0, 1, 0, 0]]
    L2 = [[1, 2, 0, 0], [2, 3, 0, 0]]
    A1 = [[2, 1, 1, 0], [0, 1, 0, 1]]
    A2 = [[-1, 2, 1, 0], [1, 2, 0, 1]]
    basis = [[2, 3], [2, 3]]
    c0 = [np.array(x) for x in c0]
    b0 = np.array(b0)
    b = [np.array(x) for x in b]
    L = [L1, L2]
    A = [A1, A2]
    L = [np.array(x) for x in L]
    A = [np.array(x) for x in A]
    basis = [np.array(x) for x in basis]
    print("\nTest Dantzig Wolfe 0")
    print(simplex_dantzig_wolfe(c0, L, b0, A, b, basis=basis, debug=True))


def test_dwp1_dantzig():
    c0 = [[-1, -2, 0, 0], [-4, -3, 0, 0]]
    b0 = [4, 3]
    b = [[4, 2], [2, 5]]
    L1 = [[1, 1, 0, 0], [0, 1, 0, 0]]
    L2 = [[2, 0, 0, 0], [1, 1, 0, 0]]
    A1 = [[2, 1, 1, 0], [1, 1, 0, 1]]
    A2 = [[1, 1, 1, 0], [3, 2, 0, 1]]
    basis = [[2, 3], [2, 3]]
    c0 = [np.array(x) for x in c0]
    b0 = np.array(b0)
    b = [np.array(x) for x in b]
    L = [L1, L2]
    A = [A1, A2]
    L = [np.array(x) for x in L]
    A = [np.array(x) for x in A]
    basis = [np.array(x) for x in basis]
    print("\nTest Dantzig Wolfe 1")
    print(simplex_dantzig_wolfe(c0, L, b0, A, b, basis=basis, debug=True))


def test_reduce_equation():
    print("\nTest reduce equation")
    c = [4, 3, 2, 1]
    A = [[2, 3, 4, 4], [1, 1, 2, 1]]
    b = [6, 3]
    c = np.array(c)
    A = np.array(A)
    b = np.array(b)
    print("Primal Problem")
    print(linprog(c, A_eq=A, b_eq=b))
    row, col = A.shape
    c_s = np.concatenate((np.zeros(col), np.ones(row)))
    A_s = np.concatenate((A, np.eye(row)), axis=1)
    basis = range(col, col + row)
    opt = simplex_revised(c_s, A_s, b, basis, ret_lu=True)
    basis = opt.basis

    null_row, null_var = find_null_variable(opt.basis, A, opt.x_basis, lu_basis=opt.lu_basis)
    c_res, A_res, b_res, basis_res = reduce_equation(null_row, null_var, c, A, b, basis)
    print("\nReduce Problem")
    print("A\n%s\n%s" % (str(A), str(A_res)))
    print("b\n%s\n%s" % (str(b), str(b_res)))
    print("c\n%s\n%s" % (str(c), str(c_res)))
    print("basis\n%s\n%s" % (str(basis), str(basis_res)))
    print(simplex_revised(c_res, A_res, b_res, basis_res))


if __name__ == "__main__":
    test_revised()
    test_dual0()
    test_dual1()
    test_dwp0_revised()
    test_dwp0_dantzig()
    test_dwp1_dantzig()
    test_reduce_equation()
