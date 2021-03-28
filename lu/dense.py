from lu.utils import *


def pivot_max_row(mat, row_idxs, col_idxs, k):
    n = len(mat)
    mr = k
    mv = abs(mat[row_idxs[k]][k])
    for i in range(k + 1, n):
        cv = abs(mat[row_idxs[i]][k])
        if mv < cv:
            mr = i
            mv = cv
    return mr, k


def pivot_max_root(mat, row_idxs, col_idxs, k):
    n = len(mat)
    rk = row_idxs[k]
    ck = col_idxs[k]
    mr = k
    mv = abs(mat[rk][ck])
    for i in range(k + 1, n):
        cv = abs(mat[row_idxs[i]][ck])
        if mv < cv:
            mr = i
            mv = cv
    mc = k
    for i in range(k + 1, n):
        cv = abs(mat[rk][col_idxs[i]])
        if mv < cv:
            mc = i
            mv = cv
    if mc > k:
        mr = k
    return mr, mc


def compute_lu(mat, func_piv=pivot_max_row):
    n = len(mat)
    p = np.arange(n)
    q = np.arange(n)
    row_idxs = np.arange(n)
    col_idxs = np.arange(n)

    def pivot(k):
        return func_piv(mat, row_idxs, col_idxs, k)

    for k in range(n):
        sr, sc = pivot(k)
        p[k] = sr
        q[k] = sc
        swap(row_idxs, k, sr)
        swap(col_idxs, k, sc)
        pr, pc = row_idxs[k], col_idxs[k]
        # lower
        for i in range(k + 1, n):
            ri = row_idxs[i]
            mat[ri][pc] /= mat[pr][pc]
        # upper
        for j in range(k + 1, n):
            cj = col_idxs[j]
            for i in range(k + 1, n):
                ri = row_idxs[i]
                mat[ri][cj] -= mat[ri][pc] * mat[pr][cj]
    return mat[row_idxs].T[col_idxs].T, p, q


def ftrans_lower(mat, x):
    n = len(x)
    for i in range(n):
        for j in range(i):
            x[i] -= x[j] * mat[i][j]
    return x


def btrans_lower(mat, x):
    n = len(x)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            x[i] -= x[j] * mat[j][i]
    return x


def ftrans_upper(mat, x):
    n = len(x)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            x[i] -= x[j] * mat[i][j]
        x[i] /= mat[i][i]
    return x


def btrans_upper(mat, x):
    n = len(x)
    for i in range(n):
        for j in range(i):
            x[i] -= x[j] * mat[j][i]
        x[i] /= mat[i][i]
    return x


class LADense(object):
    def __init__(self):
        self.n = 0
        self.lu = None
        self.p = None
        self.q = None

    def factor(self, mat):
        self.n = len(mat)
        self.lu, self.p, self.q = compute_lu(mat, func_piv=pivot_max_root)

    def ftrans(self, x):
        perm_arr(x, self.p)
        ftrans_lower(self.lu, x)
        ftrans_upper(self.lu, x)
        inv_perm_arr(x, self.q)
        return x

    def btrans(self, x):
        perm_arr(x, self.q)
        btrans_upper(self.lu, x)
        btrans_lower(self.lu, x)
        inv_perm_arr(x, self.p)
        return x
