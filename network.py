# encode: utf8
import sys

class Arc(object):
    def __init__(self, aid, s, d, cost, capacity=1e16, artificial=False):
        self.aid = aid
        self.s = s ## start
        self.d = d ## end
        self.cost = cost
        self.capacity = capacity
        self.is_artifical = artificial
        ## flow info
        self.flow = 0
        self.reduced_cost = None
        self.state = -1 ## Lower -1 Tree 0 Upper 1
        self.direction = False ## False Down; True Up

    def __eq__(self, arc):
        return self.s == arc.s and self.d == arc.d

    def __str__(self):
        #string = "ArcId: %d (%d %d) Cost %d Flow %d" % (self.aid, self.s, self.d, self.cost, self.flow)
        string = "ArcId: %d (%d %d) Cost %d Flow %d Dir %s Red %s" % \
                 (self.aid, self.s, self.d, self.cost, self.flow, str(self.direction), str(self.reduced_cost))
        return string
    
    def get_neighbor(self, nid):
        if self.s == nid:
            return self.d
        elif self.d == nid:
            return self.s
        else:
            return -1

    def is_state_tree(self):
        return self.state == 0

    def is_state_upper(self):
        return self.state == 1


class Node(object):
    def __init__(self, nid, supply):
        self.nid = nid 
        self.supply = supply
        self.arc_out = set()
        ## Spanning Tree
        self.is_root = False
        self.is_leaf = False
        self.neighbor = set() ## neighbor or chlidren
        self.pred = -1
        self.num_succ = None
        ## XTI
        self.depth = None
        self.thread = None
        ## XPI
        self.son = -1
        self.brother = -1
        ## Price
        self.price = None
        self.mark = None

    def __eq__(self, node):
        return self.nid == node.nid

    def __str__(self):
        string = "NodeId: %d Supply %d Price %s Pred %d Son %d Brother %d Depth %s" % \
                (self.nid, self.supply, self.price, self.pred, self.son, self.brother, self.depth)
        return string
    
    def add_arc_out(self, aid):
        self.arc_out.add(aid)

    def add_neighbor(self, nid):
        self.neighbor.add(nid)

    def set_root(self):
        self.is_root = True
        self.pred = -1
        self.depth = 0
        self.brother = -1


class Network(object):
    def __init__(self):
        self.num_node = 0
        self.max_cost = 0
        self.max_capacity = int(1e16)
        self.nodes = {}
        self.arcs = {}
        self.arc_idx = {}

    def __str__(self):
        max_line = 10
        string = "%s" % self.num_node
        cnt = 0
        for nid, node in self.nodes.iteritems():
            head = "%d, %d" % (nid, node.supply)
            arcs = ["%d, %d" % (self.arcs[aid].d, self.arcs[aid].cost) for aid in node.arc_out]
            string = "%s\n%s: %s" % (string, head, "; ".join(arcs))
            cnt += 1
            if cnt >= max_line:
                break
        return string

    def reset(self):
        self.num_node = 0
        self.nodes.clear()
        self.arcs.clear()
        self.arc_idx.clear()

    def add_node(self, nid, supply=0):
        self.nodes[nid] = Node(nid, supply)

    def add_arc(self, s, d, cost, capacity):
        aid = len(self.arcs)
        self.max_cost += cost
        self.arcs[aid] = Arc(aid, s, d, cost, capacity)
        self.arc_idx[(s, d)] = aid
        self.nodes[s].add_arc_out(aid)

    def add_artificial_arc(self, s, d):
        aid = len(self.arcs)
        self.arcs[aid] = Arc(aid, s, d, self.max_cost, self.max_capacity, artificial=True)
        self.arc_idx[(s, d)] = aid
        self.nodes[s].add_arc_out(aid)

    def get_arc(self, nid_pr):
        if nid_pr in self.arc_idx:
            aid = self.arc_idx[nid_pr]
            return self.arcs.get(aid, None)
        return None

    def update_num_node(self):
        self.num_node = len(self.nodes)

    def get_path(self, src):
        path = [src.nid]
        while src.pred in self.nodes:
            path.append(src.pred)
            src = self.nodes[src.pred]
        return path


def read_network(file_nwk, nwk):
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
                    d = int(arc[0])
                    cost = int(arc[1])
                elif len(arc) == 3:
                    d = int(arc[0])
                    cost = int(arc[1])
                    capacity = int(arc[2])
                nwk.add_arc(nid, d, cost, capacity)
    return nwk



def find_shortest_path_bellman(nwk, dst):
    max_price = 1e16
    for node in nwk.nodes.itervalues():
        node.price = max_price
    dst.price = 0
    dst.pred = -1
    while True:
        has_change = False
        for arc in nwk.arcs.itervalues():
            s = nwk.nodes[arc.s] 
            if s == dst:
                continue
            d = nwk.nodes[arc.d] 
            if d.price == max_price:
                continue
            price = arc.cost + d.price
            if s.price > price:
                s.price = price
                s.pred = d.nid
                has_change = True
        if not has_change:
            break
    return nwk


def find_shortest_path_dijkstra(nwk, dst):
    max_price = 1e16
    for node in nwk.nodes.itervalues():
        node.price = max_price
    dst.price = 0
    dst.pred = -1
    node_shortest = set()
    while True:
        ## find current min node
        min_price = max_price
        min_node = None
        for nid, node in nwk.nodes.iteritems():
            if nid in node_shortest:
                continue
            if node.price < min_price:
                min_price = node.price
                min_node = node.nid
        if min_node is None:
            break
        node_shortest.add(min_node)
        ## update price
        for nid, node in nwk.nodes.iteritems():
            if nid in node_shortest:
                continue
            for aid in node.arc_out:
                arc = nwk.arcs[aid]
                d = nwk.nodes[arc.d]
                if d.price == max_price:
                    continue
                price = arc.cost + d.price 
                if node.price > price:
                    node.pred = arc.d
                    node.price = price
    return nwk

