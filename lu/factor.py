from scipy import linalg


class LAScipy:
    def __init__(self):
        self.lu = None
        self.piv = None

    def factor(self, B):
        self.lu, self.piv = linalg.lu_factor(B)

    def ftrans(self, x):
        x = linalg.lu_solve((self.lu, self.piv), x)
        return x

    def btrans(self, x):
        x = linalg.lu_solve((self.lu, self.piv), x, trans=1)
        return x
