# encode: utf8
import sys
import numpy as np
from scipy import sparse
from scipy import optimize

from sudoku import Sudoku

def idx2mat(idx, val, dim):
    mat = np.zeros(dim ** 2)
    idx = np.array(idx) - 1
    mat[idx] = val
    return mat.reshape(dim, dim)
    

def test_sudoku_dim2():
    print "\nTest Sudoku Constraint"
    sd2 = Sudoku()
    num_var = sd2.num_var
    pre_idx = range(1, num_var + 1)
    pre_val = [1, 2, 3, 4, 3, 4, 1, 2, 4, 3, 2, 1, 2, 1, 4, 3]
    vec_set = sd2.idx2vec(pre_idx, pre_val)
    idx, val = sd2.vec2idx(vec_set.todense())
    print "oreset index %s" % idx
    print "preset value" % val

    sdk_cnt = sd2.make_sudoku_constraint()
    mat_cnt = sd2.make_constraint_matrix(sdk_cnt)
    z = mat_cnt.dot(vec_set)
    print "sudoku number %s" % len(sdk_cnt)
    print "sudoku test %s" % all(i == 1 for i in z)
    
    mat_set = sd2.make_sudoku_matrix(pre_set=(idx, val))
    z = mat_set.dot(vec_set)
    print "preset number %s" % mat_set.shape[0]
    print "preset test %s" % all(i == 1 for i in z)


def test_sudoku_dim3():
    print "\nTest Sudoku Constraint"
    sd3 = Sudoku(3)
    num_var = sd3.num_var
    sdk_cnt = sd3.make_sudoku_constraint()
    mat_cnt = sd3.make_constraint_matrix(sdk_cnt)
    print "sudoku number %s" % len(sdk_cnt)
    print "sudoku matrix %s %s" % mat_cnt.shape


def test_sudoku_p0():
    print "\nTest Sudoku p0"
    sd2 = Sudoku()
    idx = [1, 3, 6, 8, 9, 11, 14, 16]
    val = [1, 3, 4, 2, 4, 2, 1, 3]
    mat_set = sd2.make_sudoku_matrix(pre_set=(idx, val))
    row, col = mat_set.shape
    print "size %s %s" % (row, col)
    b = np.ones(row)
    c = np.zeros(col)
    A_eq = mat_set.toarray()
    ret = optimize.linprog(c, A_eq=A_eq, b_eq=b)
    idx, val = sd2.vec2idx(ret.x)
    mat = idx2mat(idx, val, 4)
    print ret
    print "Sudoku Matrix\n%s" % str(mat)


if __name__ == "__main__":
    test_sudoku_dim2()
    test_sudoku_dim3()
    test_sudoku_p0()

