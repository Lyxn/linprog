# encode: utf8

import sys
import numpy as np

from network import *

class Cycle(object):
    def __init__(self, arc_in, node_join):
        self.arc_in = arc_in
        self.node_join = node_join
        self.arc_out = None
        self.arc_start = []
        self.arc_end = []
    
    def add_start(self, arc):
        self.arc_start.append(arc)

    def add_end(self, arc):
        self.arc_end.append(arc)

    def reset(self):
        self.arc_in = None
        self.node_join = None
        self.arc_start = []
        self.arc_end = []

    def print_cycle(self):
        print "\nCycle"
        print "Arc in: %s" % str(self.arc_in)
        print "Arc out: %s" % str(self.arc_out)
        print "Node join: %s" % str(self.node_join)
        print "Start"
        for arc in self.arc_start:
            print arc
        print "End"
        for arc in self.arc_end:
            print arc
        

class TreeAPI(Network):
    def __init__(self):
        Network.__init__(self)
        self.root_id = 0
        self.basic_arc = set()

    def init_tree_from_arc(self, basic_arc):
        self.basic_arc.update(basic_arc)
        # Init Spanning Tree 
        ## set neighbor
        for aid in self.basic_arc:
            arc = self.arcs[aid]
            arc.state = 0
            self.nodes[arc.s].add_neighbor(arc.d)
            self.nodes[arc.d].add_neighbor(arc.s)
        ## set root
        root = self.nodes[self.root_id]
        root.set_root()
        ## init tree struct
        self.init_tree_struct(root)
        ## update depth and pred
        self.update_depth(root)
        ## update thread
        self.update_thread(root)
        
        # Calculate Flow & Price
        self.calc_basic_flow(self.basic_arc)
        ## update node price
        self.calc_basic_price()
        ## update arc reduced cost
        self.calc_reduced_cost()

    def init_tree_struct(self, node):
        stk = [node]
        while len(stk) > 0:
            cur = stk.pop()
            if len(cur.neighbor) == 0:
                continue
            ## son & brother
            idx = 0
            brother = -1
            for cid in sorted(cur.neighbor):
                child = self.nodes[cid]
                child.pred = cur.nid
                child.brother = brother
                brother = cid
                if idx == 0:
                    cur.son = cid   
                ## remove parent
                if cur.nid in child.neighbor:
                    child.neighbor.remove(cur.nid)
                stk.append(child)

    def update_depth(self, node):
        stk = [node]
        while len(stk) > 0:
            cur = stk.pop()
            if cur.brother in self.nodes:
                brother = self.nodes[cur.brother]
                brother.depth = cur.depth
                stk.append(brother)
            if cur.son in self.nodes:
                son = self.nodes[cur.son]
                son.depth = cur.depth + 1
                stk.append(son)

    def update_thread(self, node):
        last_node = node
        stk = [node.brother, node.son]
        while len(stk) > 0:
            cid = stk.pop()
            if cid not in self.nodes:
                continue
            last_node.thread = cid
            last_node = self.nodes[cid]
            stk.append(last_node.brother)    
            stk.append(last_node.son)    
        last_node.thread = -1

    def calc_basic_flow(self, basic_arc):
        arc_cnt = np.zeros(self.num_node)
        node_flow = np.zeros(self.num_node)
        for aid in basic_arc:
            arc = self.arcs[aid]   
            arc_cnt[arc.s] += 1
            arc_cnt[arc.d] += 1
        for nid in range(self.num_node):
            node = self.nodes[nid]
            node_flow[nid] = node.supply
        proc_arcs = set(basic_arc)
        while True:
            if all(x != 1 for x in arc_cnt):
                break
            arc_rm = []
            for aid in proc_arcs:
                arc = self.arcs[aid]
                s = arc.s
                d = arc.d
                if arc_cnt[s] == 1:
                    arc.flow = node_flow[s]
                    node_flow[d] += arc.flow
                    arc_cnt[d] -= 1 
                    node_flow[s] = 0
                    arc_cnt[s] = 0
                    arc_rm.append(aid)
                elif arc_cnt[d] == 1:
                    arc.flow = -node_flow[d]
                    node_flow[s] -= arc.flow
                    arc_cnt[s] -= 1
                    node_flow[d] = 0
                    arc_cnt[d] = 0 
                    arc_rm.append(aid)
            for a in arc_rm:
                proc_arcs.remove(a)

    def get_basic_arc(self, pid, cid):
        arc = self.get_arc((pid, cid))
        if arc is not None and arc.is_state_tree():
            return arc
        return self.get_arc((cid, pid))

    def calc_node_price(self, cur, unk):
        arc = self.get_basic_arc(cur.nid, unk.nid)
        if arc.s == cur.nid:
            unk.price = cur.price - arc.cost
        else:
            unk.price = cur.price + arc.cost

    def calc_basic_price(self):
        root = self.nodes[self.root_id]
        root.price = 0
        stk = [root]
        while len(stk) > 0:
            cur = stk.pop()
            if cur.brother in self.nodes:
                brother = self.nodes[cur, brother]
                father = self.nodes[brother.pred]
                self.calc_node_price(father, brother)
                stk.append(brother)
            if cur.son in self.nodes:
                son = self.nodes[cur.son]
                self.calc_node_price(cur, son)
                stk.append(son)

    def calc_reduced_cost(self):
        for aid, arc in self.arcs.iteritems():
            if aid in self.basic_arc:
                arc.reduced_cost = 0
            else:
                arc.reduced_cost = arc.cost - (self.nodes[arc.s].price - self.nodes[arc.d].price)

    def sum_basic_cost(self):
        cost_sum = 0
        for aid in self.basic_arc:
            arc = self.arcs[aid]
            cost_sum += arc.cost * arc.flow
        return cost_sum

    def print_nodes(self):
        print "Node"
        for nid in self.nodes:
            print self.nodes[nid]
        
    def print_arcs(self):
        print "Basic Arc"
        for aid in self.basic_arc:
            print self.arcs[aid]
        print "NonBasis Arc"
        for aid, arc in self.arcs.iteritems():
            if not arc.is_state_tree():
                print arc

    def print_tree(self):
        print "\nTree Cost %d" % self.sum_basic_cost()
        self.print_nodes()
        self.print_arcs()

    def find_succesor(self, node):
        succesors = set()
        queue = [node.nid]
        while len(queue) > 0:
            nid = queue.pop()
            succesors.add(nid)
            node = self.nodes[nid]
            if node.son in self.nodes:
                queue.insert(0, node.son)
            if node.brother in self.nodes:
                queue.insert(0, node.brother)
        return succesors

    def find_join_node(self, a, b):
        a = self.nodes[a]
        b = self.nodes[b]
        while True:
            if a.is_root:
                return a
            elif b.is_root:
                return b
            elif a.nid == b.nid:
                return a
            if a.depth >= b.depth:
                a = self.nodes[a.pred]
            else:
                b = self.nodes[b.pred]

    def find_cycle(self, arc_in, node_join, cycle):
        arc_in.direction = True
        is_neg_dir = lambda a, b: a.s == b.s or a.d == b.d
        ## from start point to root
        cur = self.nodes[arc_in.s]
        last_arc = arc_in
        while cur.nid != node_join.nid:
            pred = self.nodes[cur.pred]
            arc = self.get_basic_arc(cur.pred, cur.nid)
            if is_neg_dir(arc, last_arc):
                arc.direction = not last_arc.direction
            else:
                arc.direction = last_arc.direction
            cycle.add_start(arc)
            last_arc = arc
            cur = pred
        ## from end point to root
        cur = self.nodes[arc_in.d]
        last_arc = arc_in
        while cur.nid != node_join.nid:
            pred = self.nodes[cur.pred]
            arc = self.get_basic_arc(cur.pred, cur.nid)
            if is_neg_dir(arc, last_arc):
                arc.direction = not last_arc.direction
            else:
                arc.direction = last_arc.direction
            last_arc = arc
            cycle.add_end(arc)
            cur = pred
        return cycle

    def find_leaving_arc(self, arc_in, cycle):
        ## update cycle direction
        anti_arcs =[]
        ## arc start
        for arc in cycle.arc_start:
            if arc.direction != arc_in.direction:
                anti_arcs.append(arc)
        ## arc end
        for arc in cycle.arc_end:
            if arc.direction != arc_in.direction:
                anti_arcs.append(arc)
        min_flow = 1e16
        min_arc = None
        for arc in anti_arcs:
            if arc.flow < min_flow:
                min_flow = arc.flow
                min_arc = arc
        cycle.arc_out = min_arc
        return min_arc, anti_arcs
    
    def update_cycle_flow(self, cycle, arc_out, anti_arcs):
        min_flow = arc_out.flow
        cycle.arc_in.flow = min_flow
        for arc in cycle.arc_end + cycle.arc_start:
            if arc in anti_arcs:
                arc.flow -= min_flow
            else:
                arc.flow += min_flow
    
    def update_node_price(self, arc_in, tree_upd):
        if arc_in.s in tree_upd:
            delta = arc_in.reduced_cost
        else:
            delta = -arc_in.reduced_cost
        for nid in tree_upd:
            self.nodes[nid].price += delta
    
    def update_reduced_cost(self, arc_in, tree_upd):
        dir_in = arc_in.s in tree_upd
        delta = arc_in.reduced_cost
        for aid, arc in self.arcs.iteritems():
            if arc.s in tree_upd and arc.d in tree_upd:
                continue
            elif arc.s not in tree_upd and arc.d not in tree_upd:
                continue
            if (arc.s in tree_upd) == dir_in:
                arc.reduced_cost -= delta
            else:
                arc.reduced_cost += delta
    
    def update_basic_arc(self, arc_in, arc_out):
        arc_in.state = 0
        arc_out.state = -1
        self.basic_arc.remove(arc_out.aid)
        self.basic_arc.add(arc_in.aid)

    def tree_add_child(self, node, child):
        ## child as son
        if node.son not in self.nodes:
            node.son = child.nid
        ## child as brother
        else:
            son = self.nodes[node.son]
            while son.brother in self.nodes:
                son = self.nodes[node.brother]
            son.brother = child.nid

    def tree_del_child(self, node, child):
        son = self.nodes[node.son]
        ## child as son
        if son.nid == child.nid:
            node.son = child.brother
            child.brother = -1
        ## child as brother
        else:
            while son.brother in self.nodes and son.brother != child.nid:
                son = self.nodes[son.brother]
            son.brother = child.brother
            child.brother = -1

    def update_tree_struct(self, arc_in, node_upd, tree_upd):
        if arc_in.s in tree_upd:
            node = self.nodes[arc_in.s]
            pred_new = self.nodes[arc_in.d]
        else:
            node = self.nodes[arc_in.d]
            pred_new = self.nodes[arc_in.s]
        node.depth = pred_new.depth + 1
        self.tree_add_child(pred_new, node)
        node_top = node
        new_pid = pred_new.nid
        #print "update tree"
        while node.nid != node_upd.nid:
            pred_old = self.nodes[node.pred] 
            ## update child
            self.tree_add_child(node, pred_old)
            ## update brother
            self.tree_del_child(pred_old, node)
            ## update predecossor
            node.pred = new_pid
            ## change node
            new_pid = node.nid
            node = pred_old
        if not node.is_root:
            self.tree_del_child(self.nodes[node.pred], node)
            node.pred = new_pid
        #self.print_nodes()
        self.update_depth(node_top)

    def simplex(self, **argv):
        max_iter = argv.get("max_iter", 1000)
        debug = argv.get("debug", False)
        for itr in range(max_iter):
            # find entering arc
            arc_in = None
            has_neg = False
            for aid, arc in self.arcs.iteritems():
                if aid in self.basic_arc:
                    continue
                if arc.reduced_cost < 0:
                    has_neg = True
                    arc_in = arc
                    break
            if not has_neg:
                sys.stderr.write("Probelm solved.\n")
                return self.basic_arc
            # find leaving arc
            node_join = self.find_join_node(arc_in.s, arc_in.d)
            ## cycle 
            cycle = Cycle(arc_in, node_join)
            cycle = self.find_cycle(arc_in, node_join, cycle)
            arc_out, anti_arcs = self.find_leaving_arc(arc_in, cycle)
            if debug:
                print "\nIteration %d" % itr
                cycle.print_cycle()
            self.update_cycle_flow(cycle, arc_out, anti_arcs)
            ## Tree update
            src_out = self.nodes[arc_out.s]
            dst_out = self.nodes[arc_out.d]
            if src_out.pred == dst_out.nid:
                node_upd = src_out
            else:
                node_upd = dst_out
            tree_upd = self.find_succesor(node_upd)
            ## Update price
            self.update_node_price(arc_in, tree_upd)
            self.update_reduced_cost(arc_in, tree_upd)
            self.update_tree_struct(arc_in, node_upd, tree_upd)
            self.update_basic_arc(arc_in, arc_out)
            if debug:
                self.print_tree()

