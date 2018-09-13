#encode=utf8
import numpy as np

from simplex import simplex_revised
from simplex import simplex_dual
from decomposition import simplex_dantzig_wolfe
from linprog import linprog
from linprog import linprog_primal


def test_revised():
    c = [-3, -1, -3]
    b = [2, 5, 6]
    A = [[2, 1, 1], [1, 2, 3], [2, 2, 1]]
    basis = [3, 4, 5]
    c = np.array(c)
    A = np.array(A)
    b = np.array(b)
    print "\nTest linprog"
    print linprog(c, A_ub=A, b_ub=b)
    basis = np.array(basis)
    num_slack = 3
    c = np.concatenate((c, np.zeros(num_slack)))
    A = np.concatenate((A, np.eye(num_slack)), axis=1)
    print "\nTest Revised Simplex"
    print simplex_revised(c, A, b, basis, debug=True)
    print "\nTest Two Phrase Method"
    print linprog_primal(c, A, b, debug=True)


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
    print "\nTest Dual Simplex"
    print simplex_dual(c, -A, -b, basis, debug=True)


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
    print "\nTest Dual Simplex"
    print simplex_dual(c, A, b, basis, debug=True, max_iter=10)


def test_dwp0_revised():
    c = [-4, -1, -3, -2]
    b = [6, 4, 5, 1, 2, 6]
    A = [[2, 2, 1, 2], [0, 1, 2, 3], [2, 1, 0, 0], [0, 1, 0, 0], [0, 0, -1, 2], [0, 0, 1, 2]]
    c = np.array(c)
    A = np.array(A)
    b = np.array(b)
    print "\nTest Dantzig Wolfe 0"
    print linprog(c, A_ub=A, b_ub=b, debug=True)


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
    print "\nTest Dantzig Wolfe 0"
    print simplex_dantzig_wolfe(c0, L, b0, A, b, basis=basis, debug=True)


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
    print "\nTest Dantzig Wolfe 1"
    print simplex_dantzig_wolfe(c0, L, b0, A, b, basis=basis, debug=True)


if __name__ == "__main__":
    test_revised()
    test_dual0()
    test_dual1()
    test_dwp0_revised()
    test_dwp0_dantzig()
    test_dwp1_dantzig()

