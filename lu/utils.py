import numpy as np


def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


def piv2idx(piv):
    n = len(piv)
    idx = np.arange(n)
    for i in range(n):
        swap(idx, i, piv[i])
    return idx


def inv_idx(idx):
    n = len(idx)
    inv = np.arange(n)
    for i in range(n):
        inv[idx[i]] = i
    return inv


def perm_mat(mat, rows, cols):
    return mat[rows].T[cols].T


def perm_arr(x, p):
    for i in range(len(p)):
        swap(x, i, p[i])


def inv_perm_arr(x, p):
    for i in range(len(p) - 1, -1, -1):
        swap(x, i, p[i])


def copy_arr(src, dst):
    for i in range(len(src)):
        dst[i] = src[i]
