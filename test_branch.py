# encode: utf8
import sys
import numpy as np
from linprog import form_standard
from linprog import linprog_primal
from branch_cut import branch_bound
from branch_cut import proc_gomory_cut

def test_branch_bound():
    c = np.array([1, 2, 0])
    a = np.array([[-4, -2, 1]])
    b = np.array([-5])
    basis = [2]
    print "Test Branch Bound"
    ret = branch_bound(c, a, b, basis, debug=True)
    nid, tree = ret
    node = tree[nid]
    print "z_opt\t%s" % node.z_opt
    print "x_opt\t%s" % str(node.x_opt)
    print "lambda\t%s" % str(node.lmbd_opt)
    print "lower\t%s" % str(node.lower)
    print "upper\t%s" % str(node.upper)


def test_gomory_cut():
    c = np.array([0, -1])
    A_ub = np.array([[3, 2], [-3, 2]])
    b_ub = np.array([6, 0])
    c, A, b = form_standard(c, A_ub=A_ub, b_ub=b_ub)
    basis = [2, 3]
    basis, x_opt, lmbd_opt = linprog_primal(c, A, b)
    print "Test Gomory Cut"
    print "init basis\t%s" % str(basis)
    print "init x_opt\t%s" % str(x_opt)
    print "init z_opt\t%s" % np.dot(c, x_opt)
    int_idx = [0, 1]
    ret = proc_gomory_cut(c, A, b, basis, int_idx=int_idx, debug=True)
    print ret


if __name__ == "__main__":
    #test_branch_bound()
    test_gomory_cut()

