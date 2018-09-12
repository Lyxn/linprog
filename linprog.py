# encode: utf8
import sys
import numpy as np
from scipy import linalg
from simplex import simplex_revised


def form_standard(c, A_eq=None, b_eq=None, A_ub=None, b_ub=None, lower={}, upper={}, **argv):
    """
    Convert the linear program to standard form
    Input:
    Return:
        Success: (c_tot, A_tot, b_tot)
        Fail:
        -1: illegal
    """
    # init
    debug = argv.get("debug", False)
    # check 
    if (A_eq is not None and b_eq is None) \
       or (A_ub is not None and b_ub is None):
        sys.stder.write("Problme illegal\n")
        return -1
    # Problem size
    num_var = len(c)
    num_eq = A_eq.shape[0] if A_eq is not None else 0
    num_ub = A_ub.shape[0] if A_ub is not None else 0
    num_lower = len(lower)
    num_upper = len(upper)
    num_slack = num_ub + num_lower + num_upper
    row_tot = num_eq + num_slack
    col_tot = num_var + num_slack
    A_var = []
    b_tot = []
    ### equality
    if A_eq is not None:
        A_var.append(A_eq)
        b_tot.append(b_eq)
    ### inequality
    if A_ub is not None:
        A_var.append(A_ub)
        b_tot.append(b_ub)
    ### lower bounds
    eye_var = np.eye(num_var)
    if len(lower) > 0:
        lower_idx = sorted(lower.keys())
        b0 = -np.array([lower[i] for i in lower_idx])
        A0 = -eye_var.take(lower_idx, axis=0)
        A_var.append(A0)
        b_tot.append(b0)
    ### upper bounds
    eye_var = np.eye(num_var)
    if len(upper) > 0:
        upper_idx = sorted(upper.keys())
        b0 = np.array([upper[i] for i in upper_idx])
        A0 = eye_var.take(upper_idx, axis=0)
        A_var.append(A0)
        b_tot.append(b0)
    b_tot = np.concatenate(b_tot)
    A_var = np.concatenate(A_var)
    A_slack = np.concatenate((np.zeros((num_eq, num_slack)), np.eye(num_slack)))
    A_tot = np.concatenate((A_var, A_slack), axis=1)
    c_tot = np.concatenate((c, np.zeros(num_slack)))
    return c_tot, A_tot, b_tot


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
        sys.stderr.write("Problem invalid\n")
        return -1
    basis, x0, _ = ret
    if not all(is_zero(i) for i in x0[col:]):
        sys.stderr.write("Problem infeasilble\n")
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


def linprog_primal(c, A, b, **argv):
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
        sys.stderr.write("Problem infeasible\n")
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
        sys.stderr.write("Problem unsolved\n")
        return ret1
    basis, x_opt, lmbd_opt = ret1
    if debug:
        print "\nPrimal Problem solved"
        print "z_opt\t%s" % np.dot(x_opt, c)
        print "x_opt\t%s" % str(x_opt)
        print "lambda_opt\t%s" % str(lmbd_opt)
    return basis, x_opt, lmbd_opt


def linprog(c, **argv):
    c_tot, A_tot, b_tot = form_standard(c, **argv)
    return linprog_primal(c_tot, A_tot, b_tot, **argv)

