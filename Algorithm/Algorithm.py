import copy
import numpy as np
from numpy import linalg
from Matrix.InputSystem import InputSystem


class Algorithm:

    eqSystem: InputSystem
    Xk: np.array
    Q: np.array
    maxK: int
    requiredPrecision: float

    def __init__(self, maxK: int, requiredPrecision: float):
        self.maxK = maxK
        self.requiredPrecision = requiredPrecision

    def init(self, eqSystem: InputSystem):
        np.seterr(all='raise')
        self.eqSystem = eqSystem
        self.Xk = [[0] for i in range(self.eqSystem.n)]

    def core(self):
        qInv: np.array = linalg.inv(self.Q)
        for i in range(1, self.maxK+1):
            try:
                qInvQA = np.matmul(qInv, (self.Q - self.eqSystem.A))
                self.Xk = np.matmul(qInvQA, self.Xk) + np.matmul(qInv, self.eqSystem.b)
                Rk = np.matmul(self.eqSystem.A, self.Xk) - self.eqSystem.b
                precision: float = linalg.norm(Rk) / linalg.norm(self.eqSystem.b)
                if precision < self.requiredPrecision:
                    print('Step: {}'.format(i))
                    print('Precision: {}'.format(precision))
                    return self.Xk
            except FloatingPointError:
                print('Solution overflow after {} steps.'.format(i))
                return None
        print('Solution not found after {} steps.'.format(self.maxK))
        return None

    def JacobiMatrix(self):
        self.Q = copy.deepcopy(self.eqSystem.D)

    def GaussSeidelMatrix(self):
        self.Q = self.eqSystem.D + self.eqSystem.L

    def Jacobi(self, eqSystem: InputSystem):
        print('Jacobi')
        self.init(eqSystem)
        self.JacobiMatrix()
        return self.core()

    def GaussSeidel(self, eqSystem: InputSystem):
        print('GaussSeidel')
        self.init(eqSystem)
        self.GaussSeidelMatrix()
        return self.core()
