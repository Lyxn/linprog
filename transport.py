# encode: utf8
from __future__ import print_function

import sys
from collections import defaultdict

from base import Optimum
from utils import *


def dct2mat(dct, size):
    mat = np.zeros(size)
    for i, cols in dct.items():
        for j, val in cols.items():
            mat[i, j] = val
    return mat


def northwest_corner(sources, sinks):
    basis = defaultdict(dict)
    m = len(sources)
    n = len(sinks)
    i = 0
    j = 0
    cur_src = sources[0]
    cur_snk = sinks[0]
    while True:
        if cur_src < cur_snk:
            basis[i][j] = cur_src
            cur_snk -= cur_src
            i += 1
            if i < m:
                cur_src = sources[i]
            else:
                break
        else:
            basis[i][j] = cur_snk
            cur_src -= cur_snk
            j += 1
            if j < n:
                cur_snk = sinks[j]
            else:
                break
    return basis


def transpose(basis):
    trn = defaultdict(dict)
    for i, row in basis.items():
        for j, val in row.items():
            trn[j][i] = val
    return trn


def calc_dual_variable(basis, cost):
    m, n = cost.shape
    row_val = np.zeros(m)
    col_val = np.zeros(n)
    row_res = set(range(m))
    col_res = set(range(n))
    # col_res.remove(0)
    row_res.remove(0)
    while True:
        for i in basis:
            for j in basis[i]:
                if i in row_res and j not in col_res:
                    row_val[i] = cost[i][j] - col_val[j]
                    row_res.remove(i)
                elif i not in row_res and j in col_res:
                    col_val[j] = cost[i][j] - row_val[i]
                    col_res.remove(j)
        if len(col_res) == 0 and len(row_res) == 0:
            break
    return row_val, col_val


def calc_optimum(basis, cost):
    opt = 0
    for i in basis:
        for j in basis[i]:
            opt += cost[i][j] * basis[i][j]
    return opt


def has_idx(mat, i, j):
    return i in mat and j in mat[i]


def is_unsigned_one(i, basis, basis_sgn):
    row_var = basis[i]
    row_sgn = basis_sgn[i]
    return len(row_var) - len(row_sgn) == 1


def count_basis(basis, size):
    row = np.zeros(size[0])
    col = np.zeros(size[1])
    for i in basis:
        for j in basis[i]:
            row[i] += 1
            col[j] += 1
    return row, col


def update_basis(basis, neg_idx, size, debug=False):
    is_zero_all = lambda arr: all(x == 0 for x in arr)
    m, n = size
    basis_sgn = defaultdict(dict)
    neg_x, neg_y = neg_idx
    row_sgn = np.zeros(m)
    col_sgn = np.zeros(n)
    row_sgn[neg_x] = 1
    col_sgn[neg_y] = 1
    ## count element of basis
    row_num, col_num = count_basis(basis, size)
    if debug:
        print("neg_idx\t%s %s" % neg_idx)
        print("row_num\t%s" % str(row_num))
        print("col_num\t%s" % str(col_num))
    ## cycle
    for itr in range(m + n):
        for i in basis:
            for j in basis[i]:
                ## signed element
                if has_idx(basis_sgn, i, j):
                    continue
                ## one unsigned element in row
                elif row_num[i] == 1:
                    sgn = -row_sgn[i]
                    basis_sgn[i][j] = sgn
                    col_sgn[j] += sgn
                    row_sgn[i] = 0
                    row_num[i] -= 1
                    col_num[j] -= 1
                ## one unsigned element in column
                elif col_num[j] == 1:
                    sgn = -col_sgn[j]
                    basis_sgn[i][j] = sgn
                    row_sgn[i] += sgn
                    col_sgn[j] = 0
                    row_num[i] -= 1
                    col_num[j] -= 1
        if debug:
            print("sgn mat\n%s" % str(dct2mat(basis_sgn, size)))
            print("sgn row\t%s" % str(row_sgn))
            print("sgn col\t%s" % str(col_sgn))
            print("num row\t%s" % str(row_num))
            print("num col\t%s" % str(col_num))
        ## cycle complete
        if is_zero_all(col_sgn) and is_zero_all(row_sgn):
            break
    ## min value
    min_val = 1e16
    min_idx = None
    for i in basis:
        for j in basis[i]:
            if basis_sgn[i][j] == -1 and basis[i][j] < min_val:
                min_val = basis[i][j]
                min_idx = (i, j)
    ## updpat basis
    min_x, min_y = min_idx
    basis[min_x].pop(min_y)
    for i in basis:
        for j in basis[i]:
            basis[i][j] += basis_sgn[i][j] * min_val
    basis[neg_x][neg_y] = min_val


def transport(sources, sinks, cost, **argv):
    ## argument
    max_iter = argv.get("max_iter", 100)
    debug = argv.get("debug", False)
    ## size
    size = cost.shape
    m, n = cost.shape
    ## basis solution
    basis = northwest_corner(sources, sinks)
    ## iteration
    for itr in range(max_iter):
        opt0 = calc_optimum(basis, cost)
        ## dual variable
        dual_row, dual_col = calc_dual_variable(basis, cost)
        if debug:
            print("\nIteration %s" % itr)
            print("optimum\t%s" % opt0)
            print("Basis\n%s" % str(dct2mat(basis, size)))
            print("dual row\t%s" % str(dual_row))
            print("dual col\t%s" % str(dual_col))
        ## reduced cost
        has_neg_rdc = False
        neg_idx = None
        for i in range(m):
            for j in range(n):
                rdc = cost[i][j] - dual_row[i] - dual_col[j]
                if is_neg(rdc):
                    has_neg_rdc = True
                    neg_idx = (i, j)
                    break
            if has_neg_rdc:
                break
        if not has_neg_rdc:
            sys.stderr.write("Problem Solved\n")
            return Optimum(z_opt=opt0, basis=basis, num_iter=itr)
        if debug:
            print("neg_idx\t%s %s" % neg_idx)
        ## update basis
        update_basis(basis, neg_idx, size=size)

    sys.stderr.write("Iteration exceed %s\n" % max_iter)
    sys.stderr.write("Current optimum %s\n" % opt0)
    return Optimum(status=1, z_opt=opt0, basis=basis, num_iter=max_iter)
