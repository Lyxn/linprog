# encode: utf8
import numpy as np
from scipy import linalg


class EtaCol(object):
    def __init__(self, eta, p):
        self.eta = eta
        self.p = p
        self.n = len(eta)

    def ftrans(self, x):
        x[self.p] /= self.eta[self.p]
        for i in range(self.n):
            if i != self.p:
                x[i] -= x[self.p] * self.eta[i]
        return x

    def btrans(self, x):
        y = 0
        for i in range(self.n):
            if i != self.p:
                y += x[i] * self.eta[i]
        x[self.p] = (x[self.p] - y) / self.eta[self.p]
        return x


class PF(object):
    def __init__(self):
        self.lu = None
        self.piv = None
        self.factors = []

    def invert(self, B):
        self.lu, self.piv = linalg.lu_factor(B)
        self.factors = []

    def ftrans(self, x):
        x = linalg.lu_solve((self.lu, self.piv), x)
        for es in self.factors:
            x = es.ftrans(x)
        return x

    def btrans(self, x):
        ns = len(self.factors)
        x = np.array(x)
        for i in range(ns - 1, -1, -1):
            x = self.factors[i].btrans(x)
        x = linalg.lu_solve((self.lu, self.piv), x, trans=1)
        return x

    def update(self, eta, p):
        es = EtaCol(eta, p)
        self.factors.append(es)
