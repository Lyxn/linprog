# encode: utf8
from __future__ import print_function

from graph.transport import *


def test_northwest_corner():
    print("\nTest Northwest Corner 1")
    sources = [30, 80, 10, 60]
    sinks = [10, 50, 20, 80, 20]
    m = len(sources)
    n = len(sinks)
    basis = northwest_corner(sources, sinks)
    print(dct2mat(basis, (m, n)))

    print("\nTest Northwest Corner 2")
    sources = [30, 40, 20, 60]
    sinks = [50, 20, 40, 40]
    m = len(sources)
    n = len(sinks)
    basis = northwest_corner(sources, sinks)
    print(dct2mat(basis, (m, n)))


def test_dual():
    print("\nTest Dual Variable")
    sources = [30, 80, 10, 60]
    sinks = [10, 50, 20, 80, 20]
    cost = [3, 4, 6, 8, 9, 2, 2, 4, 5, 5, 2, 2, 2, 3, 2, 3, 3, 2, 4, 2]
    m = len(sources)
    n = len(sinks)
    cost = np.reshape(cost, (m, n))
    basis = northwest_corner(sources, sinks)
    rows, cols = calc_dual_variable(basis, cost)
    print("Cost\n%s" % str(cost))
    print("Basis\n%s" % str(dct2mat(basis, (m, n))))
    print("dual row\t%s" % str(rows))
    print("dual col\t%s" % str(cols))
    cst_rdc = cost.T - rows
    cst_rdc = cst_rdc.T - cols
    print("Reduced Cost\n%s" % str(cst_rdc))


def test_update_basis():
    print("\nTest Update Basis")
    sources = [30, 80, 10, 60]
    sinks = [10, 50, 20, 80, 20]
    cost = [3, 4, 6, 8, 9, 2, 2, 4, 5, 5, 2, 2, 2, 3, 2, 3, 3, 2, 4, 2]
    m = len(sources)
    n = len(sinks)
    size = (m, n)
    cost = np.reshape(cost, (m, n))
    basis = northwest_corner(sources, sinks)
    print("Basis init\n%s" % str(dct2mat(basis, (m, n))))
    neg_idx = (0, 4)
    update_basis(basis, neg_idx, size, debug=True)
    print("Basis update\n%s" % str(dct2mat(basis, (m, n))))


def test_transport1():
    print("\nTest Transport Problem 1")
    sources = [30, 80, 10, 60]
    sinks = [10, 50, 20, 80, 20]
    cost = [3, 4, 6, 8, 9, 2, 2, 4, 5, 5, 2, 2, 2, 3, 2, 3, 3, 2, 4, 2]
    m = len(sources)
    n = len(sinks)
    cost = np.reshape(cost, (m, n))
    opt = transport(sources, sinks, cost, debug=True)
    print(opt)

    print("\nTest Transport Problem 1")
    sources = [30, 80, 10, 20]
    sinks = [10, 50, 20, 40, 20]
    cost = [5, 7, 6, 8, 9, 2, 2, 4, 5, 5, 2, 2, 2, 3, 2, 3, 3, 2, 4, 2]
    cost = np.reshape(cost, (m, n))
    opt = transport(sources, sinks, cost, debug=True)


def test_transport2():
    print("\nTest Transport Problem 2")
    sources = [25, 25, 50]
    sinks = [15, 20, 30, 35]
    cost = [10, 5, 6, 7, 8, 2, 7, 6, 9, 3, 4, 8]
    m = len(sources)
    n = len(sinks)
    cost = np.reshape(cost, (m, n))
    opt = transport(sources, sinks, cost, debug=True)
    print(opt)


def test_transport3():
    print("\nTest Transport Problem 3")
    sources = [7, 11, 18, 16]
    sinks = [10, 27, 15]
    M = 1e9
    cost = [5, 6, M, 8, 4, 3, M, 9, M, M, 3, 6]
    m = len(sources)
    n = len(sinks)
    cost = np.reshape(cost, (m, n))
    opt = transport(sources, sinks, cost, debug=True)
    print(opt)


if __name__ == "__main__":
    test_northwest_corner()
    test_dual()
    test_update_basis()
    test_transport1()
    test_transport2()
    test_transport3()
