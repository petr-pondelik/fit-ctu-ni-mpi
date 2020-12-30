import copy
import numpy as np
from numpy import linalg
from Matrix.InputSystem import InputSystem
from Matrix.Norm import Norm


class Algorithm:

    eqSystem: InputSystem
    Xk_1: np.array
    Xk: np.array
    Q: np.array
    maxK: int
    requiredPrecision: float

    def init(self, eqSystem: InputSystem):
        print('Init')
        self.eqSystem = eqSystem
        self.maxK = 1000
        self.requiredPrecision = 0.000001
        self.Xk_1 = [[0] for i in range(self.eqSystem.n)]

    def core(self):
        print('{}: {}'.format(0, self.Xk_1))
        qInv: np.array = linalg.inv(self.Q)
        for i in range(1, self.maxK+1):
            qInvQA = np.matmul(qInv, (self.Q - self.eqSystem.A))
            self.Xk = np.matmul(qInvQA, self.Xk_1) + np.matmul(qInv, self.eqSystem.b)
            Rk = np.matmul(self.eqSystem.A, self.Xk) - self.eqSystem.b
            precision: float = linalg.norm(Rk) / linalg.norm(self.eqSystem.b)
            # if i < 1000:
            #     print('Residuum: {}'.format(Rk))
            #     print('Precision: {}'.format(precision))
            if precision < self.requiredPrecision:
                print(i)
                return self.Xk
            self.Xk_1 = self.Xk
        return None

    def JacobiMatrix(self):
        self.Q = copy.deepcopy(self.eqSystem.D)

    def GaussSeidelMatrix(self):
        self.Q = self.eqSystem.D + self.eqSystem.L

    def Jacobi(self, eqSystem: InputSystem):
        self.init(eqSystem)
        self.JacobiMatrix()
        return self.core()

    def GaussSeidel(self, eqSystem: InputSystem):
        self.init(eqSystem)
        self.GaussSeidelMatrix()
        return self.core()
