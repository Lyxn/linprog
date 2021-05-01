# encode: utf8
from __future__ import print_function

from graph.network_simplex import *


def test_read_network():
    print("\nTest Network")
    file_nwk = "../data/small.nwk"
    nwk = Network()
    nwk = read_network(file_nwk, nwk)
    print(nwk)

    print("\nTest Network")
    file_nwk = "../data/shortestpath.nwk"
    nwk = Network()
    nwk = read_network(file_nwk, nwk)
    print(nwk)


def test_shortest_path():
    print("\nTest Shortest Path")
    file_nwk = "../data/shortestpath.nwk"
    nwk = Network()
    nwk = read_network(file_nwk, nwk)
    src = 0
    dst = nwk.num_node - 1
    src = nwk.nodes[src]
    dst = nwk.nodes[dst]
    print("Bellman")
    nwk = find_shortest_path_bellman(nwk, dst)
    print(src)
    print("Path %s" % str(nwk.get_path(src)))
    print("Dijkstra")
    nwk = find_shortest_path_dijkstra(nwk, dst)
    print(src)
    print("Path %s" % str(nwk.get_path(src)))


def test_init_tree():
    print("\nTest Init Tree")
    file_nwk = "../data/test.nwk"
    nwk = TreeAPI()
    read_network(file_nwk, nwk)
    arc_pr = [(3, 0), (3, 2), (1, 2), (4, 1)]
    basic_arc = [nwk.arc_idx[pr] for pr in arc_pr]
    nwk.init_tree_from_arc(basic_arc)
    nwk.print_tree()


def test_network_simplex0():
    print("\nTest Network Simplex 0")
    file_nwk = "../data/test.nwk"
    nwk = TreeAPI()
    read_network(file_nwk, nwk)
    arc_pr = [(3, 0), (3, 2), (1, 2), (4, 1)]
    basic_arc = [nwk.arc_idx[pr] for pr in arc_pr]
    nwk.init_tree_from_arc(basic_arc)
    nwk.simplex(debug=True)


def test_network_simplex1():
    print("\nTest Network Simplex 1")
    file_nwk = "../data/small.nwk"
    nwk = TreeAPI()
    read_network(file_nwk, nwk)
    arc_pr = [(4, 0), (0, 2), (3, 0), (2, 1)]
    basic_arc = [nwk.arc_idx[pr] for pr in arc_pr]
    nwk.init_tree_from_arc(basic_arc)
    print("Init Tree")
    nwk.print_tree()
    nwk.simplex(debug=True)


def test_network_simplex_artificial():
    print("\nTest Network Simplex Artificial")
    # file_nwk = "./data/test.nwk"
    # file_nwk = "./data/small.nwk"
    # file_nwk = "./data/shortestpath.nwk"
    fmt_name = lambda x: "../data/%s.nwk" % x
    nwk_name = ["test", "small", "shortestpath"]
    nwk_list = [fmt_name(x) for x in nwk_name]
    for file_nwk in nwk_list:
        nwk = TreeAPI()
        read_network(file_nwk, nwk)
        nwk.init_artificial_tree()
        nwk.simplex()
        nwk.print_optimum()


if __name__ == "__main__":
    test_read_network()
    test_shortest_path()
    test_init_tree()
    test_network_simplex0()
    test_network_simplex1()
    test_network_simplex_artificial()
