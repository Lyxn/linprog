# encode: utf8
import numpy as np

from branch_cut import branch_bound
from branch_cut import proc_gomory_cut
from linprog import form_standard
from linprog import linprog_primal


def test_branch_bound():
    print("\nTest Branch Bound")
    c = np.array([1, 2, 0])
    a = np.array([[-4, -2, 1]])
    b = np.array([-5])
    basis = [2]
    ret = branch_bound(c, a, b, basis, debug=True, deep=False)
    nid, tree = ret
    node = tree[nid]
    print("z_opt\t%s" % node.z_opt)
    print("x_opt\t%s" % str(node.x_opt))
    print("lower\t%s" % str(node.lower))
    print("upper\t%s" % str(node.upper))


def test_gomory_cut():
    print("\nTest Gomory Cut")
    c = np.array([0, -1])
    A_ub = np.array([[3, 2], [-3, 2]])

    b_ub = np.array([6, 0])
    print("Init")
    c, A, b = form_standard(c, A_ub=A_ub, b_ub=b_ub)
    opt = linprog_primal(c, A, b)
    print(opt)
    basis = opt.basis
    x_basis = opt.x_basis
    lu_basis = opt.lu_basis
    int_idx = [0, 1]
    ret = proc_gomory_cut(c, A, b, basis, int_idx=int_idx, debug=True, x_basis=x_basis, lu_basis=lu_basis)
    print(ret[0])


if __name__ == "__main__":
    test_branch_bound()
    test_gomory_cut()
