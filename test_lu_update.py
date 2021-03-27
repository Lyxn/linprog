# encode: utf8
from __future__ import print_function

import numpy as np
from scipy import linalg

from lu.ft_update import calc_lu_lower
from lu.ft_update import lu_update_col
from lu.pf_update import PF
from lu.utils import inv_idx
from lu.utils import piv2idx


def test_piv2idx():
    print("\nTEST PIVOT")
    # piv
    piv = [2, 2, 2]
    idxs = [2, 0, 1]
    print("Lapack\t%s\nPermutation\t%s" % (piv, piv2idx(piv)))
    assert np.allclose(piv2idx(piv), idxs)
    # piv
    piv = [2, 2, 3, 3]
    idxs = [2, 0, 3, 1]
    print("Lapack\t%s\nPermutation\t%s" % (piv, piv2idx(piv)))
    assert np.allclose(piv2idx(piv), idxs)


def test_lu_update_col():
    size = 5
    B = np.reshape(range(size ** 2), (size, size)) + np.eye(size)
    Bt = B.T
    aq = np.zeros(size)
    aq[0] = 1
    lu, piv = linalg.lu_factor(B)
    piv_py = piv2idx(piv)
    piv_inv = inv_idx(piv_py)
    out = 1
    B_new = np.row_stack((Bt[0:out], Bt[out + 1:], aq)).T
    ret = lu_update_col(lu, out, aq[piv_py])
    H, pivs, trns = ret
    U = np.triu(H)
    L = calc_lu_lower(lu, out, trns)
    B_upd = L.dot(U)[piv_inv]
    print("\nTEST LU UPDATE")
    print("piv_lapack\t%s" % str(piv))
    print("piv_py\t%s" % str(piv_py))
    print("H\n%s" % str(H))
    print("update piv\t%s" % str(pivs))
    print("update trans\t%s" % str(trns))
    print("L\n%s" % str(L))
    print("B_new\n%s" % str(B_new))
    print("B_upd\n%s" % str(B_upd))
    assert np.allclose(B_new, B_upd)


def test_pf_update():
    size = 5
    B0 = np.reshape(range(size ** 2), (size, size)) + np.eye(size)
    Bt = B0.T
    eye = np.eye(size)
    aq = eye[0]
    pf = PF()
    pf.factor(B0)
    af0 = np.copy(aq)
    af0 = pf.ftrans(af0)
    af1 = linalg.solve(B0, aq)
    assert np.allclose(af0, af1)
    p = 4
    B1 = np.row_stack((Bt[0:p], aq, Bt[p + 1:])).T
    # print("B0\n%s" % str(B0))
    # print("B1\n%s" % str(B1))
    pf.update(af0, p)
    ee = np.arange(0, size, dtype=float)
    print("Forward")
    ef0 = np.copy(ee)
    ef0 = pf.ftrans(ef0)
    ef1 = linalg.solve(B1, ee)
    assert np.allclose(ef0, ef1)
    print("ef0\t%s\nef1\t%s" % (str(ef0), str(ef1)))
    print("Backward")
    eb0 = np.copy(ee)
    eb0 = pf.btrans(eb0)
    eb1 = linalg.solve(B1.T, ee)
    assert np.allclose(eb0, eb1)
    print("eb0\t%s\neb1\t%s" % (str(eb0), str(eb1)))


if __name__ == "__main__":
    # test_lu_update_col()
    test_pf_update()
