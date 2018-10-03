# encode: utf8

import sys

from network import *

def test_read_network():
    print "\nTest Network"
    file_nwk = "./data/small.nwk"
    nwk = read_network(file_nwk)
    print nwk

    print "\nTest Network"
    file_nwk = "./data/shortestpath.nwk"
    nwk = read_network(file_nwk)
    print nwk


if __name__ == "__main__":
    test_read_network()
