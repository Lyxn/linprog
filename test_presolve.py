import numpy as np

from linprog import find_null_variable
from linprog import linprog
from linprog import reduce_equation
from linprog import simplex_revised


def test_reduce_equation():
    print("\nTest reduce equation")
    c = [4, 3, 2, 1]
    A = [[2, 3, 4, 4], [1, 1, 2, 1]]
    b = [6, 3]
    c = np.array(c)
    A = np.array(A)
    b = np.array(b)
    print("Primal Problem")
    # TODO fix infeasible
    ret = linprog(c, A_eq=A, b_eq=b)
    assert type(ret) == int
    row, col = A.shape
    c_s = np.concatenate((np.zeros(col), np.ones(row)))
    A_s = np.concatenate((A, np.eye(row)), axis=1)
    basis = range(col, col + row)
    opt = simplex_revised(c_s, A_s, b, basis, ret_lu=True)
    assert type(opt) != int
    basis = opt.basis

    null_row, null_var = find_null_variable(opt.basis, A, opt.x_basis, lu_factor=opt.lu_factor)
    c_res, A_res, b_res, basis_res = reduce_equation(null_row, null_var, c, A, b, basis)
    print("\nReduce Problem")
    print("A\n%s\n%s" % (str(A), str(A_res)))
    print("b\n%s\n%s" % (str(b), str(b_res)))
    print("c\n%s\n%s" % (str(c), str(c_res)))
    print("basis\n%s\n%s" % (str(basis), str(basis_res)))
    ret = simplex_revised(c_res, A_res, b_res, basis_res)
    assert type(ret) != int


if __name__ == '__main__':
    test_reduce_equation()
