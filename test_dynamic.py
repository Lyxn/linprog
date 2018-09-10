# encode: utf8
import sys

from dynamic import *

def test_knapsack01():
    c = [2, 1, 10, 2, 4]
    a = [2, 1, 4, 1, 12]
    b = 15
    print "0-1 knapsack"
    mat = knapsack01(c, a, b)
    x = back_path(a, mat)
    print "mat\n%s" % str(mat)
    print "x\t%s" % str(x)
    print "value\t%s" % np.dot(x, c)

    print "Perfect knapsack"
    mat = knapsack01_perfect(c, a, b)
    x = back_path(a, mat)
    print "mat\n%s" % str(mat)
    print "x\t%s" % str(x)
    print "value\t%s" % np.dot(x, c)


if __name__ == "__main__":
    test_knapsack01()

