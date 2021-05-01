# encode: utf8
import numpy as np

from lu.dense import LADense
from lu.factor import LAScipy


class EtaCol:
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


class PF:
    def __init__(self, la=LADense, debug=False):
        if la is None:
            la = LAScipy
        self.lu_factor = la()
        self.factors = []
        self.debug = debug
        if debug:
            self.bench_la = LAScipy()

    def factor(self, B):
        B1 = np.copy(B)
        self.lu_factor.factor(B1)
        self.factors = []
        if self.debug:
            self.bench_la.factor(B)

    def ftrans(self, x):
        x = self.lu_factor.ftrans(x)
        for es in self.factors:
            x = es.ftrans(x)
        return x

    def btrans(self, x):
        ns = len(self.factors)
        for i in range(ns - 1, -1, -1):
            x = self.factors[i].btrans(x)
        x = self.lu_factor.btrans(x)
        return x

    def update(self, eta, p):
        es = EtaCol(eta, p)
        self.factors.append(es)
