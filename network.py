# encode: utf8

import sys
import numpy as np

from base import Optimum
from utils import *


class Arc(object):
    def __init__(self, aid, start, end, cost, capacity=1e16):
        self.aid = aid
        self.start = start
        self.end = end
        self.cost = cost
        self.capacity = capacity
        ## flow info
        self.flow = None
        self.reduced_cost = None
        self.state = None


class Node(object):
    def __init__(self, nid, supply):
        self.nid = nid 
        self.supply = supply
        self.arc_out = []
        ## Spanning Tree
        self.is_root = False
        self.is_leaf = False
        self.pred = None
        self.depth = None
        self.thread = None
        self.next = None
        ## Price
        self.price = None


class Network(object):
    def __init__(self):
        self.num_node = None
        self.nodes = {}
        self.arcs = {}

    def add_node(self, nid, supply=0):
        self.nodes[nid] = Node(nid, supply)

    def add_arc(self, start, end, cost, capacity):
        aid = len(self.arcs)
        self.arcs[aid] = Arc(aid, start, end, cost, capacity)
        self.nodes[start].arc_out.append(aid)

    def __str__(self):
        max_line = 10
        string = "%s" % self.num_node
        cnt = 0
        for nid, node in self.nodes.iteritems():
            head = "%d, %d" % (nid, node.supply)
            arcs = ["%d, %d" % (self.arcs[aid].end, self.arcs[aid].cost) for aid in node.arc_out]
            string = "%s\n%s: %s" % (string, head, "; ".join(arcs))
            cnt += 1
            if cnt >= max_line:
                break
        return string


def read_network(file_nwk):
    nwk = Network()
    MAX_CAPACITY = 1e16
    with open(file_nwk, "r") as f:
        line = f.readline().strip()
        nwk.num_node = int(line) if line.isdigit() else -1
        for line in f:
            node, arcs = line.strip().split(":")
            nd = node.split(",")
            nid = int(nd[0])
            supply = int(nd[1]) if len(nd) == 2 else 0
            nwk.add_node(nid, supply)
            for arc in arcs.split(";"):
                arc = arc.split(",")
                capacity = MAX_CAPACITY
                if len(arc) == 2:
                    end = int(arc[0])
                    cost = int(arc[1])
                elif len(arc) == 3:
                    end = int(arc[0])
                    cost = int(arc[1])
                    capacity = int(arc[2])
                nwk.add_arc(nid, end, cost, capacity)
    return nwk

