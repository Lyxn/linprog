# encode: utf8
from __future__ import print_function

import time

import numpy as np
from scipy import optimize
from scipy import sparse

from simplex import simplex_revised
from modeling.sudoku import Sudoku
from utils import is_integer_list


def test_sudoku_dim2():
    print("\nTest Sudoku Constraint")
    sd2 = Sudoku()
    num_var = sd2.num_var
    pre_idx = range(1, num_var + 1)
    pre_val = [1, 2, 3, 4, 3, 4, 1, 2, 4, 3, 2, 1, 2, 1, 4, 3]
    vec_set = sd2.idx2vec(pre_idx, pre_val)
    idx, val = sd2.vec2idx(vec_set.todense())
    print("preset index %s" % idx)
    print("preset value" % val)

    sdk_cnt = sd2.make_sudoku_constraint()
    mat_cnt = sd2.make_constraint_matrix(sdk_cnt)
    z = mat_cnt.dot(vec_set)
    print("sudoku number %s" % len(sdk_cnt))
    print("sudoku test %s" % all(i == 1 for i in z))

    mat_set = sd2.make_sudoku_matrix(pre_set=(idx, val))
    z = mat_set.dot(vec_set)
    print("preset number %s" % mat_set.shape[0])
    print("preset test %s" % all(i == 1 for i in z))


def test_sudoku_p0():
    print("\nTest Sudoku p0")
    sd2 = Sudoku()
    idx = [1, 3, 6, 8, 9, 11, 14, 16]
    val = [1, 3, 4, 2, 4, 2, 1, 3]
    idx = idx[:0]
    val = val[:0]
    mat_set = sd2.make_sudoku_matrix(pre_set=(idx, val))
    row, col = mat_set.shape
    print("size %s %s" % (row, col))
    b = np.ones(row)
    c = np.zeros(col)
    A = mat_set.toarray()
    ## scipy linprog
    cur = time.time()
    ret = optimize.linprog(c, A_eq=A, b_eq=b)
    print("\nScipy time cost %5s" % (time.time() - cur))
    mat = sd2.vec2mat(ret.x)
    print(ret)
    print("Sudoku Matrix\n%s" % str(mat))
    ## simplex LU
    num_slack = row
    c = np.concatenate((c, np.ones(row)))
    A = np.concatenate((A, np.eye(row)), axis=1)
    basis = range(col, col + row)
    cur = time.time()
    opt = simplex_revised(c, A, b, basis, eps=1e-10, max_iter=1000)
    print("\nSimplexLU time cost %5s" % (time.time() - cur))
    mat = sd2.vec2mat(opt.x_opt)
    print(opt)
    print("Sudoku Matrix\n%s" % str(mat))


def str2int(s):
    return [int(i) for i in s if str.isdigit(i)]


def test_sudoku_dim3():
    print("\nTest Sudoku Constraint")
    idxs = range(1, 82)
    vals = "736851249 582974613 194236875 318645927 649127538 275398164 961483752 427569381 853712496"
    vals = str2int(vals)
    sdk = Sudoku(3)
    print("Sudoku preset\n%s" % sdk.idx2mat(idxs, vals))
    num_var = sdk.num_var
    mat_cnt = sdk.make_sudoku_matrix()
    vec_set = sdk.idx2vec(idxs, vals)
    print("sudoku matrix %s %s" % mat_cnt.shape)
    print("vector sum %s" % vec_set.sum())
    z = mat_cnt.dot(vec_set)
    print("sudoku test %s" % all(i == 1 for i in z))


def test_sudoku_p1():
    print("\nTest Sudoku p0")
    sdk = Sudoku(3)
    rows = "11 22222 33333 444 5555 666 77777 88888 99"
    cols = "59 24678 34679 379 2468 137 13467 23468 15"
    vals = "59 89461 42685 897 4173 251 91437 27598 81"
    rows = str2int(rows)
    cols = str2int(cols)
    vals = str2int(vals)
    print("Sudoku preset")
    row_sp = np.array(rows) - 1
    col_sp = np.array(cols) - 1
    print(sparse.csr_matrix((vals, (row_sp, col_sp))).toarray())
    # mat_set = sdk.make_sudoku_matrix(pre_set=(rows, cols, vals))
    mat_set = sdk.make_sudoku_matrix()
    row, col = mat_set.shape
    print("size %s %s" % (row, col))
    b = np.ones(row)
    c = np.zeros(col)
    A = mat_set.toarray()
    ## scipy linprog
    cfg = {"maxiter": 10000}
    cur = time.time()
    ret = optimize.linprog(c, A_eq=A, b_eq=b, options=cfg)
    print("\nScipy time cost %5s" % (time.time() - cur))
    print(ret)
    if ret.success:
        mat = sdk.vec2mat(ret.x)
        print("Sudoku Matrix\n%s" % str(mat))
    ## simplex LU
    num_slack = row
    c_s = np.concatenate((c, np.ones(row)))
    A_s = np.concatenate((A, np.eye(row)), axis=1)
    basis = range(col, col + row)
    cur = time.time()
    opt = simplex_revised(c_s, A_s, b, basis, eps=1e-10, max_iter=10000, ret_lu=True)
    print("\nSimplexLU time cost %5s" % (time.time() - cur))
    print(opt)
    if opt.x_opt is not None:
        mat = sdk.vec2mat(opt.x_opt[:col])
        print("x_opt sum %s integer %s" % (opt.x_opt.sum(), is_integer_list(opt.x_opt, 1e-6)))
        print("Sudoku Matrix\n%s" % str(mat))
    int_idx = range(col)


if __name__ == "__main__":
    test_sudoku_dim2()
    # test_sudoku_dim3()
    test_sudoku_p0()
    # test_sudoku_p1()
