# encode: utf8
from __future__ import print_function

import numpy as np


def knapsack01(c, a, b):
    """
    0-1 Knapsack Problem
    min c * x
    s.t a * x <= b, 
        x_i = {0, 1},
    where c_i, a_i, b is all positive integer
    """
    n = len(c)
    mat = np.zeros((n + 1, b + 1))
    for i in range(1, n + 1):
        for j in range(1, b + 1):
            if a[i - 1] > j:
                mat[i][j] = mat[i - 1][j]
            elif mat[i - 1][j] >= mat[i - 1][j - a[i - 1]] + c[i - 1]:
                mat[i][j] = mat[i - 1][j]
            else:
                mat[i][j] = mat[i - 1][j - a[i - 1]] + c[i - 1]
    return mat


def back_path(a, mat):
    print(a)
    row = mat.shape[0] - 1
    col = mat.shape[1] - 1
    x = np.zeros(row)
    while row > 0:
        print(row, col, mat[row][col])
        if mat[row][col] == mat[row - 1][col]:
            x[row - 1] = 0
            row = row - 1
        else:
            x[row - 1] = 1
            col = col - a[row - 1]
            row = row - 1
        print(row, col, mat[row][col])
    return x


def knapsack01_perfect(c, a, b):
    """
    0-1 Knapsack Problem with equality constraint
    min c * x
    s.t a * x = b, 
        x_i = {0, 1},
    where c_i, a_i, b is all positive integer
    """
    n = len(c)
    mat = np.zeros((n + 1, b + 1))
    for i in range(1, n + 1):
        for j in range(1, b + 1):
            if a[i - 1] > j:
                mat[i][j] = mat[i - 1][j]
            elif mat[i - 1][j] >= mat[i - 1][j - a[i - 1]] + c[i - 1]:
                mat[i][j] = mat[i - 1][j]
            else:
                # last_weight = 0 if mat[i-1][j - a[i-1]] == 0 else j - a[i-1]
                # if last_weight + a[i-1] != j:
                #    mat[i][j] = mat[i-1][j]
                # else:
                #    mat[i][j] = mat[i-1][j - a[i-1]] + c[i-1]
                if mat[i - 1][j - a[i - 1]] != 0 or a[i - 1] == j:
                    mat[i][j] = mat[i - 1][j - a[i - 1]] + c[i - 1]
                else:
                    mat[i][j] = mat[i - 1][j]
    return mat
