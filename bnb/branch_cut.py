# encode: utf8
from __future__ import print_function

import sys

import numpy as np

from linprog import form_standard
from lu.factor import LAScipy
from simplex import simplex_dual
from utils import floor_residue
from utils import get_unit_vector
from utils import is_integer


class Node(object):
    """ Node of Branch
    Attribute:
        nid: node id
        pid: parent id
        children: the set of children
        branch_type: 0: bound 1: cut
        lower: lower bounds
        upper: lower bounds
        cut_set: the set of cut constrants
        num_var: number of raw variables
        num_slack: number of slack variables
        basis: the basis of the current problem
        basis_raw: the basis of the raw problem
        x_opt: optimum variables
        z_opt: optimum object
        lmbd_opt: optimum dual variables 
        cut_active: active cut constraints
        is_solved: whether or not the subproblem has been solved
        is_int: whether or not the subproblem has a integer solution
        status: problem status
    """

    def __init__(self, nid, pid=0, **argv):
        self.nid = nid
        self.pid = pid
        self.children = set()
        self.branch_type = argv.get("branch_type", 0)
        self.lower = dict(argv.get("lower", {}))
        self.upper = dict(argv.get("upper", {}))
        self.cut_set = argv.get("cut_set", set())
        self.num_var = argv.get("num_var", 0)
        self.num_slack = len(self.lower) + len(self.upper) + len(self.cut_set)
        self.basis = []
        self.basis_raw = argv.get("basis_raw", [])
        self.x_opt = []
        self.z_opt = []
        self.lmbd_opt = []
        self.cut_active = []
        self.is_solved = False
        self.is_int = False
        self.status = 0

    def form_program(self, c_raw, A_raw, b_raw):
        ret = form_standard(c_raw, A_eq=A_raw, b_eq=b_raw, lower=self.lower, upper=self.upper)
        return ret

    def solve(self, c_raw, A_raw, b_raw, **argv):
        """ Solve subproblem
        Return:
            0: success
            -1: illegal 
            -2: problme unsolvable
        """
        # argv
        debug = argv.get("debug", False)
        # check
        if len(self.basis_raw) != len(b_raw):
            sys.stderr.write("Basic solution invalid\n")
            return -1
        # standard form
        ret_form = self.form_program(c_raw, A_raw, b_raw)
        if type(ret_form) == int:
            sys.stderr.write("Standard form invalid\n")
            return -1
        c_tot, A_tot, b_tot = ret_form
        num_var = len(c_raw)
        num_raw = len(b_raw)
        self.num_var = num_var
        basis_tot = self.basis_raw + list(range(num_var, num_var + self.num_slack))
        if debug:
            print("\nSubproblem")
            print("num_var\t%s" % num_var)
            print("c_tot\t%s" % str(c_tot))
            print("b_tot\t%s" % str(b_tot))
            print("A_tot\n%s" % str(A_tot))
        opt = simplex_dual(c_tot, A_tot, b_tot, basis_tot)
        if type(opt) == int:
            sys.stderr.write("Problem unsolvable\n")
            return -2
        self.is_solved = True
        self.basis = opt.basis
        self.x_opt = opt.x_opt
        self.lmbd_opt = opt.lmbd_opt
        self.z_opt = opt.z_opt
        self.basis_raw = self.basis[:num_raw]
        return 0

    @staticmethod
    def is_integer_solution(basis, x_opt, int_idx):
        for i in basis:
            if i >= len(x_opt):
                print("Integer fail %s %s %s" % (i, str(x_opt), str(basis)))
            if i in int_idx and not is_integer(x_opt[i]):
                return False
        return True

    def process(self, c_raw, A_raw, b_raw, int_idx, **argv):
        self.status = self.solve(c_raw, A_raw, b_raw, **argv)
        if self.is_solved:
            self.is_int = self.is_integer_solution(self.basis, self.x_opt, int_idx)


def branch_bound(c, A_eq, b_eq, basis, int_idx=None, **argv):
    """
    Branch algorithm for integer linear programming
    Return:
        node: the optimum node
        -1: illegal
    """
    ## argv
    max_iter = argv.get("max_iter", 100)
    max_iter_sub = argv.get("max_iter", 10000)
    debug = argv.get("debug", False)
    deep_first = argv.get("deep", True)
    num_var = len(c)
    if int_idx is None:
        int_idx = range(num_var)
    tree_dict = {}
    root_id = 0
    root = Node(0, basis_raw=basis)
    tree_dict[root_id] = root
    node_stack = [root_id]
    opt_val = 1e16
    opt_nid = 0
    active_cut_tot = {}
    if debug:
        print("\nInit")
        print("num_var\t%s" % num_var)
        print("int_idx\t%s" % str(int_idx))
    for itr in range(max_iter):
        if len(node_stack) == 0:
            return opt_nid, tree_dict
        nid = node_stack.pop()
        if nid not in tree_dict:
            return -1
        node = tree_dict[nid]
        if debug:
            print("\nIteration %s" % itr)
            print("nid\t%s" % nid)
            print("basis pre\t%s" % node.basis_raw)
        ret = node.process(c, A_eq, b_eq, int_idx=int_idx, max_iter=max_iter_sub)
        if debug:
            print("Node")
            print("status\t%s" % node.status)
            print("z\t%s" % node.z_opt)
            print("x\t%s" % node.x_opt)
            print("basis pro\t%s" % node.basis_raw)
        ## Pruning
        if node.status < 0:
            sys.stderr.write("SubProblem unsolvable\n")
            continue
        if node.z_opt >= opt_val:
            sys.stderr.write("SubProblem optimum %s over the best solution %s\n" % (node.z_opt, opt_val))
            continue
        if node.is_int:
            sys.stderr.write("SubProblem %s has integer solution %s, optimum %s\n" % (nid, node.x_opt, node.z_opt))
            if node.z_opt < opt_val:
                opt_nid = nid
                opt_val = node.z_opt
            continue
        ## Branch
        cut_idx = 0
        var_idx = None
        b_val = None
        for i in node.basis:
            if not is_integer(node.x_opt[i]) and i in int_idx:
                var_idx = i
                b_val = node.x_opt[i]
                break
            cut_idx += 1
        ### upper bound
        upper = {}
        upper.update(node.upper)
        upper[var_idx] = np.floor(b_val)
        nid_ub = len(tree_dict)
        node_ub = Node(nid_ub, pid=nid, basis_raw=node.basis_raw, lower=node.lower, upper=upper)
        tree_dict[nid_ub] = node_ub
        ### lower bound
        lower = {}
        lower.update(node.lower)
        lower[var_idx] = np.ceil(b_val)
        nid_lb = len(tree_dict)
        node_lb = Node(nid_lb, pid=nid, basis_raw=node.basis_raw, lower=lower, upper=node.upper)
        tree_dict[nid_lb] = node_lb
        ### push stack
        if not deep_first:
            node_stack.append(nid_ub)
            node_stack.append(nid_lb)
        else:
            node_stack.append(nid_lb)
            node_stack.append(nid_ub)

        if debug:
            print("Branch")
            print("var\t%s" % var_idx)
            print("val\t%s" % b_val)
            print("stack\t%s" % str(node_stack))
    return opt_nid, tree_dict


def get_gomory_cut(A, x_basis, basis, cut_idx, **argv):
    # argv
    lu_factor = argv.get("lu_factor")
    # init
    row, col = A.shape
    nonbasis = [i for i in range(col) if i not in basis]
    B = A.take(basis, axis=1)
    D = A.take(nonbasis, axis=1)
    if lu_factor is None:
        lu_factor = LAScipy()
        lu_factor.factor(B)
    # calc y_cut
    e_c = get_unit_vector(row, cut_idx)
    u_c = lu_factor.btrans(e_c)
    y_c = D.T.dot(u_c)
    b_cut = -floor_residue(x_basis[cut_idx])
    y_res = [-floor_residue(y) for y in y_c]
    y_cut = np.zeros(col)
    y_cut[nonbasis] = y_res
    return (y_cut, b_cut)


def proc_gomory_cut(c, A, b, basis, **argv):
    # argv 
    max_iter = argv.get("max_iter", 100)
    max_iter_sub = argv.get("max_iter_sub", 10000)
    debug = argv.get("debug", False)
    int_idx = argv.get("int_idx", None)
    lu_factor = argv.get("lu_factor")
    x_basis = argv.get("x_basis")
    opt = None
    # init
    row_raw = len(b)
    col_raw = len(c)
    if int_idx is None:
        int_idx = set(range(col_raw))
    c_tot = c
    A_tot = A
    b_tot = b
    cut_pool = {}
    for itr in range(max_iter):
        row, col = A_tot.shape
        nonbasis = [i for i in range(col) if i not in basis]
        if lu_factor is None:
            lu_factor = LAScipy()
            B = A_tot.take(basis, axis=1)
            lu_factor.factor(B)
        if x_basis is None:
            x_basis = lu_factor.ftrans(b_tot)
        cut_idx = None
        cut_val = None
        for i in range(row):
            idx = basis[i]
            val = x_basis[i]
            if idx in int_idx and not is_integer(val):
                cut_idx = i
                cut_val = val
        if cut_idx is None:
            sys.stderr.write("Problem solved\n")
            return opt, cut_pool
        ## calc cut
        if debug:
            print("\nIteration %s" % itr)
            print("size\t%s %s" % (row, col))
            print("basis\t%s" % str(basis))
            print("x_basis\t%s" % x_basis)
            print("cut_idx\t%s" % cut_idx)
        y_cut, b_cut = get_gomory_cut(A_tot, x_basis, basis, cut_idx, lu_factor=lu_factor)
        cid = len(cut_pool)
        cut_pool[cid] = (y_cut, b_cut)
        ## add cut
        c_tot = np.concatenate((c_tot, [0]))
        A_tot = np.concatenate((A_tot, np.zeros((row, 1))), axis=1)
        y_tot = np.concatenate((y_cut, [1]))
        A_tot = np.concatenate((A_tot, [y_tot]))
        b_tot = np.concatenate((b_tot, [b_cut]))
        basis.append(col)
        if debug:
            print("cut yl\t%s" % str(y_cut))
            print("cut y0\t%s" % b_cut)
            print("basis\t%s" % str(basis))
            print("c_tot\t%s" % str(c_tot))
            print("b_tot\t%s" % str(b_tot))
        opt = simplex_dual(c_tot, A_tot, b_tot, basis, ret_lu=True, max_iter=max_iter_sub)
        if type(opt) == int:
            sys.stderr.write("Problem unsolvable\n")
            return -1
        basis = opt.basis
        x_basis = opt.x_basis
        lu_factor = opt.lu_factor
        if debug:
            print(opt)

    return opt, cut_pool
