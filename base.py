# encode: utf8
import sys

class Optimum(object):
    def __init__(self, **argv):
        self.status = 0
        self.z_opt = argv.get("z_opt")
        self.x_opt = argv.get("x_opt")
        self.lmbd_opt = argv.get("lmbd_opt")
        self.basis = argv.get("basis")
        self.x_basis = argv.get("x_basis")
        self.lu_basis = argv.get("lu_basis") 
        self.inv_basis = argv.get("inv_basis")
        self.num_iter = argv.get("num_iter", 0) 
        self.num_col = len(self.x_opt) if self.x_opt is not None else 0
        self.num_row = len(self.basis) if self.basis is not None else 0

    def __str__(self):
        return "\nOptimum\t%s\nnum_iter\t%s\nx_opt\t%s\nbasis\t%s\n" % (self.z_opt, self.num_iter, str(self.x_opt), str(self.basis))

