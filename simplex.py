# encode=utf8

import sys
import numpy as np
from scipy import linalg


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
    """
    # init 
    eps = argv.get("eps", 1e-16)
    max_iter = argv.get("max_iter", 100)
    debug = argv.get("debug", False)
    is_neg = lambda x: x < -eps
    is_pos = lambda x: x > eps
    is_zero = lambda x: x <= eps and x >= -eps
    # check problem 
    row, col = A.shape
    if not check_size(c, A, b, basis):
        info = "Size illegal c:%s A:%s,%s b:%s, basis:%s\n\n" % (len(c), row, col, len(b), len(basis))
        sys.stderr.write(info)
        return -1
    # check primal feasible
    if any(is_neg(i) for i in b):
        sys.stderr.write("Basis illegal b:%s" % str(b))
        return -1

    # iteration
    for itr in range(max_iter):
        nonbasis = [i for i in range(col) if i not in basis]
        B = A.take(basis, axis=1)
        c_b = c[basis]
        D = A.take(nonbasis, axis=1)
        c_d = c[nonbasis]
        # solve system B
        lu_p = linalg.lu_factor(B)
        x_b = linalg.lu_solve(lu_p, b)
        lmbd = linalg.lu_solve(lu_p, c_b, trans=1)
        r_d = c_d - D.T.dot(lmbd)
        if debug:
            print "\nIteration %d" % itr
            print "basis\t%s" % str(basis)
            print "x_b\t%s" % str(x_b)
            print "lambda\t%s" % str(lmbd)
            print "z\t%s" % np.dot(x_b, c_b)
            print "nonbasis\t%s" % str(nonbasis)
            print "r_d\t%s" % str(r_d)
        # check reduced cost
        neg_ind = [i for i in range(len(r_d)) if is_neg(r_d[i])]
        if len(neg_ind) == 0:
            sys.stderr.write("Problem solved\n\n")
            return basis, align_basis(x_b, basis, col), lmbd
        ind_new = nonbasis[neg_ind[0]]
        # pivot
        a_q = A.take(ind_new, axis=1)
        y_q = linalg.lu_solve(lu_p, a_q)
        pos_ind = [i for i in range(len(y_q)) if is_pos(y_q[i])]
        if len(pos_ind) == 0:
            sys.stderr.write("Problem unbounded\n\n")
            return -2
        ratio = [x_b[i] / y_q[i] for i in pos_ind]
        min_ind = np.argmin(ratio)
        out = pos_ind[min_ind]
        ind_out = basis[out]
        basis[out] = ind_new
        if debug:
            print "y_q\t%s" % str(y_q)
            print "basis in %s out %s" % (ind_new, ind_out)
    return basis, align_basis(x_b, basis, col), lmbd


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
    """
    # init 
    eps = argv.get("eps", 1e-16)
    max_iter = argv.get("max_iter", 100)
    debug = argv.get("debug", False)
    is_neg = lambda x: x < -eps
    is_pos = lambda x: x > eps
    is_zero = lambda x: x <= eps and x >= -eps

    # check problem 
    row, col = A.shape
    if not check_size(c, A, b, basis):
        info = "Size illegal c:%s A:%s,%s b:%s, basis:%s\n\n" % (len(c), row, col, len(b), len(basis))
        sys.stderr.write(info)
        return -1
    # check dual feasible
    if any(is_neg(i) for i in c):
        sys.stderr.write("Basis illegal b:%s" % str(c))
        return -1

    # iteration
    for itr in range(max_iter):
        nonbasis = [i for i in range(col) if i not in basis]
        B = A.take(basis, axis=1)
        c_b = c[basis]
        D = A.take(nonbasis, axis=1)
        c_d = c[nonbasis]
        # solve system B
        lu_p = linalg.lu_factor(B)
        x_b = linalg.lu_solve(lu_p, b)
        lmbd = linalg.lu_solve(lu_p, c_b, trans=1)
        r_d = c_d - D.T.dot(lmbd)
        if debug:
            print "\nIteration %d" % itr
            print "basis\t%s" % str(basis)
            print "x_b\t%s" % str(x_b)
            print "lambda\t%s" % str(lmbd)
            print "z\t%s" % np.dot(x_b, c_b)
            print "nonbasis\t%s" % str(nonbasis)
            print "r_d\t%s" % str(r_d)
        # check x_b
        neg_ind = [i for i in range(len(x_b)) if is_neg(x_b[i])]
        if len(neg_ind) == 0:
            sys.stderr.write("Problem solved\n\n")
            return basis, align_basis(x_b, basis, col), lmbd
        ind_neg = nonbasis[neg_ind[0]]
        ind_out = basis[ind_neg]
        # pivot
        e_q = np.zeros(row)
        e_q[ind_neg] = 1
        u_q = linalg.lu_solve(lu_p, e_q, trans=1)
        y_q = D.T.dot(e_q)
        y_neg = [i for i in range(len(y_q)) if is_neg(y_q[i])]
        if len(y_neg) == 0:
            sys.stderr.write("Problem unbounded\n\n")
            return -2
        ratio = [r_d[i] / -y_q[i] for i in y_neg]
        min_ind = np.argmin(ratio)
        ind_new = nonbasis[y_neg[min_ind]]
        basis[ind_neg] = ind_new
        if debug:
            print "y_q\t%s" % str(y_q)
            print "basis in %s out %s" % (ind_new, ind_out)
    return basis, align_basis(x_b, basis, col), lmbd


def init_basis_primal(A, b, **argv):
    """
    Solve Artifical Linear Programming
        min 1*s 
        s.t s + A*x = b,
            s, x >= 0
    Input:
        A: equation constraint
        b: equation constraint
        eps: tolerance
        max_iter: max number of iteration
    Return: 
        success: (basis, x, lambda)
        fail:
        -1: invalid
        -2: infeasible, the minimum is not zero
    """
    eps = argv.get("eps", 1e-16)
    is_zero = lambda x: x <= eps and x >= -eps
    row, col = A.shape
    vec_one = np.ones(row)
    Ap = np.concatenate((A, np.eye(row)), axis=1)
    cp = np.concatenate((-vec_one.dot(A), np.zeros(row)))
    basis = range(col, col + row)
    ret = simplex_revised(cp, Ap, b, basis)
    if type(ret) == int:
        sys.stderr.write("Problem invalid\n\n")
        return -1
    basis, x0, _ = ret
    if not all(is_zero(i) for i in x0[col:]):
        sys.stderr.write("Problem infeasilble\n\n")
        return -2
    return basis, x0


def check_basis_slack(basis, A, **argv):
    """ Whether the basis has slack variables 
    Input:
        basis: index of basic solution
        A: constraint matrix
        replace: whether or not replace slack variables 
    Return
        0: not slack 
        1: has slack 
    """
    row, col = A.shape
    idx_slack = [i for i in range(len(basis)) if basis[i] >= col]
    if len(idx_slack) == 0:
        return 0
    nonbasis = [i for i in range(col) if i not in basis]
    ## Replace slack with first non-basis
    ## TODO whether the basis is singular
    replace= argv.get("replace", True)
    if replace:
        j = 0
        for i in idx_slack:
            basis[i] = nonbasis[j]
            j += 1
    ## TODO Extend A and c
    return 1


def lp_primal(c, A, b, **argv):
    """
    Solve Linear Programming in standard form
        min c*x 
        s.t A*x = b,
            x >= 0
    Input:
        c: object vector
        A: equation constraint
        b: equation constraint
        eps: tolerance
        max_iter: max number of iteration
    Return: 
        success: (basis, x, lambda)
        fail:
        -1: illegal
        -2: unbounded
        -3: infeasible

    """
    # Init
    eps = argv.get("eps", 1e-16)
    debug = argv.get("debug", False)
    is_neg = lambda x: x < -eps
    # size
    row, col = A.shape
    if debug:
        print "\nProblem size row %s col %s" % (row, col)
    # Make sure b >= 0
    for i in range(row):
        if is_neg(b[i]):
            b[i] = -b[i] 
            A[i] = -A[i] 
    # Init basic solution
    ret0 = init_basis_primal(A, b)
    if type(ret0) == int:
        sys.stderr.write("Problem infeasible\n\n")
        return -3
    basis, x0 = ret0
    if debug:
        print "\nBasic Problem solved"
        print "basis\t%s" % str(basis)
        print "x0\t%s" % str(x0)
    check_basis_slack(basis, A)
    # Solve LP
    ret1 = simplex_revised(c, A, b, basis, debug=debug)
    if type(ret1) == int:
        sys.stderr.write("Problem unsolved\n\n")
        return ret1
    basis, x_opt, lmbd_opt = ret1
    if debug:
        print "\nPrimal Problem solved"
        print "z_opt\t%s" % np.dot(x_opt, c)
        print "x_opt\t%s" % str(x_opt)
        print "lambda_opt\t%s" % str(lmbd_opt)
    return basis, x_opt, lmbd_opt

