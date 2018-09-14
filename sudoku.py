# encode: utf8
import sys
import numpy as np
from scipy import sparse
 
def idx2mat(idx, val, dim):
    if len(idx) == 0:
        return np.zeros((dim, dim))
    mat = np.zeros(dim ** 2)
    idx = np.array(idx) - 1
    mat[idx] = val
    return mat.reshape(dim, dim)

    
class Sudoku(object):
    """ Sudoku Problem
    """
    def __init__(self, n=2):
        self.num = n
        self.num_dim = n ** 2
        self.num_var = self.num_dim ** 2
        self.num_bool = self.num_var * self.num_dim

    def idx2bool(self, idx, val=1):
        offset = self.num_dim * (idx - 1)
        return offset + (val - 1)

    def bool2idx(self, num):
        idx = int(num / self.num_dim + 1)
        val = int(num % self.num_dim + 1)
        return idx, val

    def mat2bool(self, row, col, val=1):
        """ Row first
        """
        dim = self.num_dim
        if row > dim or col > dim:
            sys.stderr.write("Size %s %s exceed\n" % (row, col))
            return -1
        idx = (row - 1) * dim + col   
        return self.idx2bool(idx, val)

    def vec2idx(self, vec):
        eps = 1e-16
        idx_list = []
        val_list = []
        for i in range(self.num_bool):
            if abs(vec[i] - 1) <= eps:
                idx, val = self.bool2idx(i)
                idx_list.append(idx)
                val_list.append(val)
        return idx_list, val_list

    def idx2vec(self, idx_list, val_list):
        n = len(val_list)
        idx = [self.idx2bool(idx_list[i], val_list[i]) for i in range(n)]
        vec = sparse.lil_matrix((self.num_bool, 1))
        vec[idx] = 1
        return vec
    
    def idx2mat(self, idx_list, val_list):
        return idx2mat(idx_list, val_list, self.num_dim)

    def vec2mat(self, vec):
        idx, val = self.vec2idx(vec)
        return self.idx2mat(idx, val)

    def constraint_idx(self, idxs, vals):
        return [[self.idx2bool(idxs[i], vals[i])] for i in range(len(vals))]

    def constraint_mat(self, rows, cols, vals):
        return [[self.mat2bool(rows[i], cols[i], vals[i])] for i in range(len(vals))]
        
    def constraint_var(self):
        var_cnt = []
        var_begin = 0
        for i in range(1, self.num_var+1):
            var_next = var_begin + self.num_dim
            var_cnt.append(range(var_begin, var_next))
            var_begin = var_next
        return var_cnt 

    def constraint_sum(self, cnt_idx):
        if len(cnt_idx) != self.num_dim:
            sys.stderr.write("Constraint size %s illegal\n" % len(cnt_idx))
            return -1
        cnt_begin = [self.idx2bool(i) for i in cnt_idx]
        cnt_sum = []
        for k in range(self.num_dim):
            cnt = [k + i for i in cnt_begin]
            cnt_sum.append(cnt)
        return cnt_sum

    def constraint_row(self):
        row_cnt = []
        row_begin = 1
        for i in range(self.num_dim):
            row_next = row_begin + self.num_dim
            cnt_idx = range(row_begin, row_next)
            cnt_sum = self.constraint_sum(cnt_idx)
            row_cnt += cnt_sum
            row_begin = row_next
        return row_cnt

    def constraint_col(self):
        col_cnt = []
        dim = self.num_dim
        for col in range(1, dim+1):
            cnt_idx = [col + i * dim  for i in range(dim)]
            cnt_sum = self.constraint_sum(cnt_idx)
            col_cnt += cnt_sum
        return col_cnt

    def get_block_index(self, offset):
        blk_idx = []
        num = self.num
        dim = self.num_dim
        for i in range(num):
            blk_idx += range(offset, offset+num)
            offset += dim
        return blk_idx

    def constraint_block(self):
        num = self.num
        dim = self.num_dim
        blk_cnt = []
        for i in range(num):
            for j in range(num):
                blk_offset = num * dim * i + num * j + 1
                blk_idx = self.get_block_index(blk_offset)
                blk_cnt += self.constraint_sum(blk_idx)
        return blk_cnt

    def make_constraint_matrix(self, cnts):
        row = len(cnts)
        indptr = []
        indices = []
        ptr = 0
        for i in range(row):
            cnt = cnts[i]
            indices += cnt
            indptr.append(ptr)
            ptr += len(cnt)
        indptr.append(ptr)
        data = np.ones(len(indices))
        return sparse.csr_matrix((data, indices, indptr), shape=(row, self.num_bool))

    def make_sudoku_constraint(self, **argv):
        pre_set = argv.get("pre_set", [])
        sdk_cnt = []
        sdk_cnt += self.constraint_var()
        sdk_cnt += self.constraint_row()
        sdk_cnt += self.constraint_col()
        sdk_cnt += self.constraint_block()
        if len(pre_set) == 2:
            idxs = pre_set[0]
            vals = pre_set[1]
            val_cnt = self.constraint_idx(idxs, vals)
            sdk_cnt += val_cnt
        elif len(pre_set) == 3:
            rows = pre_set[0]
            cols = pre_set[1]
            vals = pre_set[2]
            val_cnt = self.constraint_mat(rows, cols, vals)
            sdk_cnt += val_cnt
        return sdk_cnt

    def make_sudoku_matrix(self, **argv):
        sdk_cnt = self.make_sudoku_constraint(**argv)
        return self.make_constraint_matrix(sdk_cnt)

