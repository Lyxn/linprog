# encode: utf8
from scipy import linalg

from utils import *


def conv_piv(vec):
    num = len(vec)
    raw = list(range(num))
    for i in range(num):
        raw[i], raw[vec[i]] = raw[vec[i]], raw[i]
    return raw


def inv_piv(piv):
    num = len(piv)
    inv = [0] * num
    for i in range(num):
        inv[piv[i]] = i
    return inv


def swap_row(mat, k, j):
    tmp = mat.take(k, axis=0)
    mat[k] = mat[j]
    mat[j] = tmp


def calc_lu_lower(lu, out, trn_val):
    """ no pivot
    """
    size = lu.shape[0]
    Lt = np.tril(lu, -1).T + np.eye(size)
    i = 0
    for k in range(out, size - 1):
        trn = trn_val[i]
        Lt[k][k + 1:] -= Lt[k + 1][k + 1:] * trn
        i += 1
    return Lt.T


def lu_update_col(lu, out, col_in):
    """ 
    Given A = L @ U, find A_new = P @ L_new @ U_new, 
    where A_new = [a_1,..,a_k-1, a_k+1, a_m, a_new].
    
    Input:
        LU: lu matrix
        out: index of out-column 
        col_in: new column 
        
    Return:
        H: lu matrix
        piv_ind: pivot info
        trn_val: transform value
    """
    is_big_num = lambda x: x >= 1e3
    size = lu.shape[0]
    h_L = linalg.solve_triangular(lu, col_in, lower=True, unit_diagonal=True)
    Ut = np.triu(lu).T
    H = np.row_stack((Ut[0:out], Ut[out + 1:], h_L)).T
    piv_ind = []
    trn_val = []
    # print("H0\n%s" % str(H))
    for k in range(out, size - 1):
        if is_zero(H[k, k]) and is_zero(H[k + 1, k]):
            piv_ind.append(k)
            trn_val.append(0)
            continue
        if is_zero(H[k, k]) or is_big_num(H[k, k + 1] / H[k, k]):
            swap_row(H, k, k + 1)
            piv_ind.append(k + 1)
        else:
            piv_ind.append(k)
        trn = -H[k + 1, k] / H[k, k]
        # H[k+1][k:] += H[k][k:] * trn
        ## upper info
        H[k + 1][k + 1:] += H[k][k + 1:] * trn
        ## lower info
        H[k + 1, k] = trn
        trn_val.append(trn)
    return H, piv_ind, trn_val
