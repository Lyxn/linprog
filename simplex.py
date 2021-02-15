# encode=utf8
"""
## Tips
1. Algorithm will fail when the basis is singular. 
   One could increase `eps` to avoid the bad column with nearly zero cost. 

## TODO
* Support box constraint
* Reduction of inequality, find non-extremal variable
"""
from __future__ import print_function

import sys

import numpy as np

from base import Optimum
from lu.pf_update import PF


def align_basis(x_b, basis, dim):
    x0 = np.zeros(dim)
    for i in range(len(basis)):
        x0[basis[i]] = x_b[i]
    return x0


def check_size(c, A, b, basis):
    row, col = A.shape
    if row != len(b) or col != len(c) or row != len(basis):
        return False
    else:
        return True


def simplex_revised(c, A, b, basis, **argv):
    """
    Revised simplex for Linear Programming
        min c*x 
        s.t A*x = b,
            x >= 0
    Input:
        c: object vector
        A: equation constraint
        b: equation constraint
        basis: index of basis
        eps: tolerance
        max_iter: max number of iteration
    Return: 
        success: (basis, x, lambda)
        fail:
        -1: illegal
        -2: unbounded
        -3: unsolved
    """
    # argument
    eps = argv.get("eps", 1e-16)
    max_iter = argv.get("max_iter", 100)
    debug = argv.get("debug", False)
    ret_lu = argv.get("ret_lu", False)
    is_neg = lambda x: x < -eps
    is_pos = lambda x: x > eps
    is_zero = lambda x: x <= eps and x >= -eps
    # check problem 
    row, col = A.shape
    if not check_size(c, A, b, basis):
        info = "Size illegal c:%s A:%s,%s b:%s, basis:%s\n" % (len(c), row, col, len(b), len(basis))
        sys.stderr.write(info)
        return -1
    # check primal feasible
    if any(is_neg(i) for i in b):
        sys.stderr.write("Basis illegal b:%s" % str(b))
        return -1

    basis = list(basis)
    pf = PF()
    pf.invert(A.take(basis, axis=1))
    cir = 50
    # iteration
    for itr in range(max_iter):
        nonbasis = [i for i in range(col) if i not in basis]
        c_b = c[basis]
        D = A.take(nonbasis, axis=1)
        c_d = c[nonbasis]
        # solve system B
        x_b = pf.ftrans(b)
        lmbd = pf.btrans(c_b)
        r_d = c_d - lmbd.dot(D)
        z0 = np.dot(x_b, c_b)
        if debug:
            print("\nIteration %d" % itr)
            print("z\t%s" % z0)
            print("basis\t%s" % str(basis))
            print("x_b\t%s" % str(x_b))
            print("lambda\t%s" % str(lmbd))
            print("r_d\t%s" % str(r_d))
        # check reduced cost
        neg_ind = [i for i in range(len(r_d)) if is_neg(r_d[i])]
        if len(neg_ind) == 0:
            sys.stderr.write("Problem solved\n")
            x_opt = align_basis(x_b, basis, col)
            opt = Optimum(z_opt=z0, x_opt=x_opt, lmbd_opt=lmbd, basis=basis, x_basis=x_b, num_iter=itr)
            if ret_lu:
                opt.lu_basis = (pf.lu, pf.piv)
            return opt
        ind_new = nonbasis[neg_ind[0]]
        # pivot
        a_q = A.take(ind_new, axis=1)
        y_q = pf.ftrans(a_q)
        pos_ind = [i for i in range(len(y_q)) if is_pos(y_q[i])]
        if len(pos_ind) == 0:
            sys.stderr.write("Problem unbounded\n")
            return -2
        ratio = [x_b[i] / y_q[i] for i in pos_ind]
        min_ind = np.argmin(ratio)
        out = pos_ind[min_ind]
        ind_out = basis[out]
        basis[out] = ind_new
        if debug:
            print("y_q\t%s" % str(y_q))
            print("basis in %s out %s" % (ind_new, ind_out))
        if (itr + 1) % cir == 0:
            pf.invert(A.take(basis, axis=1))
        else:
            pf.update(y_q, out)
    sys.stderr.write("Iteration exceed %s\n" % max_iter)
    sys.stderr.write("Current optimum %s\n" % z0)
    return -3


def simplex_dual(c, A, b, basis, **argv):
    """
    Dual simplex for Linear Programming
        min c*x 
        s.t A*x = b,
            x >= 0
    Input:
        c: object vector
        A: equation constraint
        b: equation constraint
        basis: index of basis
        eps: tolerance
        max_iter: max number of iteration
    Return: 
        success: (basis, x, lambda)
        fail:
        -1: illegal
        -2: unbounded
        -3: unsolved
    """
    # init 
    eps = argv.get("eps", 1e-16)
    max_iter = argv.get("max_iter", 100)
    debug = argv.get("debug", False)
    ret_lu = argv.get("ret_lu", False)
    is_neg = lambda x: x < -eps
    is_pos = lambda x: x > eps
    is_zero = lambda x: x <= eps and x >= -eps

    # check problem 
    row, col = A.shape
    if not check_size(c, A, b, basis):
        info = "Size illegal c:%s A:%s,%s b:%s, basis:%s\n" % (len(c), row, col, len(b), len(basis))
        sys.stderr.write(info)
        return -1

    basis = list(basis)
    pf = PF()
    pf.invert(A.take(basis, axis=1))
    cir = 50
    # iteration
    for itr in range(max_iter):
        nonbasis = [i for i in range(col) if i not in basis]
        c_b = c[basis]
        D = A.take(nonbasis, axis=1)
        c_d = c[nonbasis]
        # solve system B
        x_b = pf.ftrans(b)
        lmbd = pf.btrans(c_b)
        r_d = c_d - lmbd.dot(D)
        z0 = np.dot(x_b, c_b)
        # check dual feasible
        if any(is_neg(i) for i in r_d):
            sys.stderr.write("Dual infeasible r_d:%s\n" % str(r_d))
            return -1
        if debug:
            print("\nIteration %d" % itr)
            print("z\t%s" % z0)
            print("basis\t%s" % str(basis))
            print("x_b\t%s" % str(x_b))
            print("lambda\t%s" % str(lmbd))
            print("r_d\t%s" % str(r_d))
        # check x_b
        neg_ind = [i for i in range(len(x_b)) if is_neg(x_b[i])]
        if len(neg_ind) == 0:
            sys.stderr.write("Problem solved\n")
            x_opt = align_basis(x_b, basis, col)
            opt = Optimum(z_opt=z0, x_opt=x_opt, lmbd_opt=lmbd, basis=basis, x_basis=x_b, num_iter=itr)
            if ret_lu:
                opt.lu_basis = (pf.lu, pf.piv)
            return opt
        out = neg_ind[0]
        ind_out = basis[out]
        # pivot
        e_q = np.zeros(row)
        e_q[out] = 1
        u_q = pf.btrans(e_q)
        y_q = D.T.dot(u_q)
        y_neg = [i for i in range(len(y_q)) if is_neg(y_q[i])]
        if len(y_neg) == 0:
            sys.stderr.write("Problem unbounded\n")
            return -2
        ratio = [r_d[i] / -y_q[i] for i in y_neg]
        min_ind = np.argmin(ratio)
        ind_new = nonbasis[y_neg[min_ind]]
        basis[out] = ind_new
        if (itr + 1) % cir == 0:
            pf.invert(A.take(basis, axis=1))
        else:
            aq = A.take(ind_new, axis=1)
            aqf = pf.ftrans(aq)
            pf.update(aqf, out)
        if debug:
            print("y_q\t%s" % str(y_q))
            print("basis in %s out %s" % (ind_new, ind_out))
    sys.stderr.write("Iteration exceed %s\n" % max_iter)
    sys.stderr.write("Current optimum %s\n" % z0)
    return -3
