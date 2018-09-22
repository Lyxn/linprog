# encode: utf8

import sys
import numpy as np
from scipy import linalg

from factorization import conv_piv
from factorization import inv_piv
from factorization import lu_update_col
from factorization import calc_lu_lower

def test_conv_piv():
    # piv
    piv = [2, 2, 2]
    piv_py = [2, 0, 1]
    print "LA\t%s\nCONV\t%s\nPY\t%s" % (str(piv), str(conv_piv(piv)), str(piv_py))
    # piv
    piv = [2, 2, 3, 3]
    piv_py = [2, 0, 3, 1]
    print "LA\t%s\nCONV\t%s\nPY\t%s" % (str(piv), str(conv_piv(piv)), str(piv_py))


def test_lu_update_col():
    size = 10
    B = np.reshape(range(size**2), (size, size)) + np.eye(size)
    Bt = B.T
    aq = np.zeros(size)
    aq[0] = 1
    lu, piv = linalg.lu_factor(B)
    piv_py = conv_piv(piv)
    piv_inv = inv_piv(piv_py)
    out = 1
    B_new = np.row_stack((Bt[0:out], Bt[out+1:], aq)).T
    ret = lu_update_col(lu, out, aq[piv_py])
    U, pivs, trns = ret
    L = calc_lu_lower(lu, out, pivs, trns)
    print "piv\t%s" % str(piv_py)
    print "piv_inv\t%s" % str(piv_inv)
    print "B_upd\n%s" % str(L.dot(U)[piv_inv])
    print "U\n%s"  % str(U)
    print "update piv\t%s" % str(pivs)
    print "update trans\t%s" % str(trns)
    print "L\n%s"  % str(L)
    print "B_upd is close to B_new.\t%s" % np.allclose(B_new[piv_py], L.dot(U))


if __name__ == "__main__":
    #test_conv_piv()
    test_lu_update_col()

