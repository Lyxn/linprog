# encode: utf8
from __future__ import print_function

import sys

import numpy as np
from scipy import linalg

from base import Optimum
from simplex import align_basis
from simplex import simplex_revised


def check_size_dantzig(c, L, b0, A, b):
    is_equal_list = lambda x, y: len(x) == len(y) and all(x[i] == y[i] for i in range(len(x)))
    n_blk = len(A)
    row_blk = [x.shape[0] for x in A]
    col_blk = [x.shape[1] for x in A]
    col_L = [x.shape[1] for x in L]
    col_c = [len(x) for x in c]
    row_b = [len(x) for x in b]
    row_lnk = len(b0)
    if n_blk != len(c) or n_blk != len(L) or n_blk != len(b) \
            or any(row_lnk != len(L[i]) for i in range(len(L))):
        return False
    elif not is_equal_list(col_blk, col_L) or not is_equal_list(col_blk, col_c) \
            or not is_equal_list(row_blk, row_b):
        return False
    else:
        return True


def cut_array(blocks, array):
    offset = 0
    vec = []
    for n in blocks:
        end = offset + n
        vec.append(array[offset:end])
        offset = end
    return vec


def solve_basis_system(A, b, basis):
    x_b = linalg.solve(A.take(basis, axis=1), b)
    return align_basis(x_b, basis, A.shape[1])


def calc_comb_extreme(extreme, comb_list, weight, dim, offset_blk):
    x_opt = np.zeros(dim)
    n_comb = len(comb_list)
    for k in range(n_comb):
        i, j = comb_list[k]
        if i < 0:
            continue
        x_ext = extreme[i][j]
        offset = offset_blk[i]
        end = offset + len(x_ext)
        x_opt[offset:end] += x_ext * weight[k]
    return x_opt


class SubSystem(object):
    def __init__(self, c, L, A, b, **argv):
        ## argv
        basis = argv.get("basis", [])
        index = argv.get("index", -1)
        offset = argv.get("offset", -1)
        num_blk = argv.get("num_nlk", -1)
        ## linear system
        self.c = c
        self.L = L
        self.A = A
        self.b = b
        self.row = A.shape[0]
        self.col = A.shape[1]
        ## global info
        self.index = index
        self.offset = offset
        self.num_blk = num_blk
        ## temporary info
        self.basis = basis
        self.x_b = np.zeros(self.col)
        self.z_opt = 1e16

    def solve_sub(self, lmbd, lmbd_i, max_upper=1e16, debug=False):
        c_sub = self.c - self.L.T.dot(lmbd)
        if debug:
            print("sub c\t%s" % str(c_sub))
            print("sub intercept\t%s" % -lmbd_i)
        ret = simplex_revised(c_sub, self.A, self.b, self.basis)
        if type(ret) == int:
            return max_upper
        else:
            basis = ret.basis
            x_b = ret.x_opt
            self.basis = basis
            self.x_b = x_b
            z_opt = np.dot(c_sub, x_b) - lmbd_i
            return z_opt

    def calc_column(self):
        return self.L.dot(self.x_b)


def simplex_dantzig_wolfe(c, L, b0, A, b, **argv):
    """
    Dantzig Wolfe Decomposition for Linear Programming
        min sum(c_i * x_i) 
        s.t sum(L_i * x_i) = b0
            A_i * x_i = b_i,
            x_i >= 0,  i = 1,...,N
    Input:
        c: object vector
        L: linking constraints
        b0: linking constraints
        A: block constraints
        b: block constraints
        basis: basis of sub-system 
        eps: tolerance
        max_iter: max number of iteration
    Return: 
        success: Optimum
        fail:
        -1: illegal
        -2: unbounded
    """
    # init 
    eps = argv.get("eps", 1e-16)
    max_iter = argv.get("max_iter", 20)
    basis = argv.get("basis", None)
    x0 = argv.get("x_init", None)
    debug = argv.get("debug", False)
    is_neg = lambda x: x < -eps
    is_pos = lambda x: x > eps
    is_zero = lambda x: x <= eps and x >= -eps
    is_neg_any = lambda x: any(is_neg(i) for i in x)

    # check problem 
    if not check_size_dantzig(c, L, b0, A, b):
        info = "Size illegal c:%d L:%s b0:%s, A:%s b:%s\n" % (len(c), len(L), len(b0), len(A), len(b))
        sys.stderr.write(info)
        return -1
    # check feasible
    if is_neg_any(b0) or any(is_neg_any(i) for i in b):
        sys.stderr.write("Basis infeasible b0:%s b:%s\n" % (str(b0), str(b)))
        return -1

    # problem size
    n_blk = len(A)
    row_blk = [x.shape[0] for x in A]
    col_blk = [x.shape[1] for x in A]
    col_tot = sum(col_blk)
    row_lnk = len(b0)
    offset_blk = [0]
    for i in range(n_blk - 1):
        offset = offset_blk[-1] + col_blk[i]
        offset_blk.append(offset)
    c_tot = np.concatenate(c)
    ## sub problem
    sub_sys = {}
    for i in range(n_blk):
        sub_sys[i] = SubSystem(c[i], L[i], A[i], b[i], basis=basis[i], index=i, num_blk=n_blk)
    if x0 is None:
        x0 = [solve_basis_system(A[i], b[i], basis[i]) for i in range(n_blk)]
    if debug:
        print("Init")
        print("Block num\t%s" % n_blk)
        print("Block row\t%s" % str(row_blk))
        print("Block col\t%s" % str(col_blk))
        print("Block offset\t%s" % str(offset_blk))
        print("Block x0\t%s" % str(x0))
        print("Block col_tot\t%s" % col_tot)
        print("Link num\t%s" % row_lnk)

    # initialization
    ## master problem, size row_lnk+n_blk
    ### object vector
    p_zero = np.zeros(n_blk)
    p_init = np.array([np.dot(c[i], x0[i]) for i in range(n_blk)])
    p_master = np.concatenate((p_zero, p_init))
    ### constraint rhs vector
    g_master = np.concatenate((b0, np.ones(n_blk)))
    ### constraint lhs matrix 
    q_slack = np.concatenate((np.eye(row_lnk), np.zeros((n_blk, row_lnk))))
    q_basis = []
    for i in range(n_blk):
        qi = L[i].dot(x0[i])
        # print("Li\t%s" % str(L[i]))
        # print("xi\t%s" % str(x0[i]))
        # print("qi\t%s" % str(qi))
        ei = np.zeros(n_blk)
        ei[i] = 1
        qi = np.concatenate((qi, ei))
        q_basis.append(qi)
    q_basis = np.array(q_basis)
    Bt_master = np.concatenate((q_slack, q_basis.T), axis=1).T
    ### feasible x
    # x_master = linalg.solve(Bt_master, g_master, transposed=True)
    ### lagrange multiplier
    # lmbd_master = linalg.solve(Bt_master, p_master)
    ## extreme point
    extreme = dict((i, [x0[i]]) for i in range(n_blk))
    basis_zero = [(-1, -1) for i in range(row_lnk)]
    basis_init = [(i, 0) for i in range(n_blk)]
    basis_master = basis_zero + basis_init
    if debug:
        print("Master g\t%s" % str(g_master))
        print("Master p\t%s" % str(p_master))
        print("Master Q\n%s" % str(Bt_master.T))
        print("Master extreme\t%s" % str(extreme))
        print("Master basis\t%s" % str(basis_master))

    # iteration
    for itr in range(max_iter):
        ## solve basis system 
        lu_p = linalg.lu_factor(Bt_master.T)
        x_master = linalg.lu_solve(lu_p, g_master)
        lmbd_master = linalg.lu_solve(lu_p, p_master, trans=1)
        if debug:
            print("\nIteration %d" % itr)
            print("Master z\t%s" % np.dot(x_master, p_master))
            print("Master basis\t%s" % str(basis_master))
            print("Master B\n%s" % str(Bt_master.T))
            print("Master x\t%s" % str(x_master))
            print("Master p\t%s" % str(p_master))
            print("Master lambda\t%s" % str(lmbd_master))
            # x_prime = calc_comb_extreme(extreme, basis_master, x_master, col_tot, offset_blk)
            # print("Prime x\t%s" % str(x_prime))
        ## solve sub system
        lmbd0 = lmbd_master[0:row_lnk]
        lmbd1 = lmbd_master[row_lnk:]
        min_idx = None
        min_sub = 0
        for i in range(n_blk):
            z_sub = sub_sys[i].solve_sub(lmbd0, lmbd1[i], debug=debug)
            if is_neg(z_sub) and z_sub < min_sub:
                min_sub = z_sub
                min_idx = i
        if min_idx is None:
            sys.stderr.write("Problem solved\n")
            x_opt = calc_comb_extreme(extreme, basis_master, x_master, col_tot, offset_blk)
            z_opt = x_opt.dot(c_tot)
            return Optimum(z_opt=z_opt, x_opt=x_opt, num_iter=itr)
        ## update master problem
        x_new = np.copy(sub_sys[min_idx].x_b)
        q_new = L[min_idx].dot(x_new)
        e_new = np.zeros(n_blk)
        e_new[min_idx] = 1
        q_new = np.concatenate((q_new, e_new))
        y_new = linalg.lu_solve(lu_p, q_new)
        if debug:
            print("Sub index\t%s" % min_idx)
            print("Sub z\t%s" % min_sub)
            print("Sub x\t%s" % str(x_new))
            print("Master q_new\t%s" % str(q_new))
            print("Master y_new\t%s" % str(y_new))
        pos_ind = [i for i in range(len(y_new)) if is_pos(y_new[i])]
        if len(pos_ind) == 0:
            sys.stderr.write("Problem unbounded\n")
            return -2
        ratio = [x_master[i] / y_new[i] for i in pos_ind]
        min_ind = np.argmin(ratio)
        out = pos_ind[min_ind]
        basis_master[out] = (min_idx, len(extreme[min_idx]))
        Bt_master[out] = q_new
        extreme[min_idx].append(x_new)
        p_master[out] = np.dot(c[min_idx], x_new)
    sys.stderr.write("Iteration exceed %s\n" % max_iter)
    x_opt = calc_comb_extreme(extreme, basis_master, x_master, col_tot, offset_blk)
    z_opt = x_opt.dot(c_tot)
    return Optimum(z_opt=z_opt, x_opt=x_opt, num_iter=itr)
