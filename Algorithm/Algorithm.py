import copy
import numpy as np
from numpy import linalg
from Matrix.InputSystem import InputSystem


# Class providing solution of linear equations systems by implemented iteration methods
class Algorithm:

    eqSystem: InputSystem
    Xk: np.array
    Q: np.array
    maxK: int
    requiredPrecision: float

    # Configure the algorithm
    def __init__(self, maxK: int, requiredPrecision: float):
        self.maxK = maxK
        self.requiredPrecision = requiredPrecision

    # Pass the equations system to algorithm and set X0 to Xk
    def init(self, eqSystem: InputSystem):
        # Allow numpy to raise exceptions
        np.seterr(all='raise')
        self.eqSystem = eqSystem
        self.Xk = [[0] for i in range(self.eqSystem.n)]

    # Core algorithm loop
    def core(self):
        # Get inversion of the method's step matrix
        qInv: np.array = linalg.inv(self.Q)
        for i in range(1, self.maxK+1):
            try:
                # Perform Q^-1 * (Q - A) multiplication
                qInvQA = np.matmul(qInv, (self.Q - self.eqSystem.A))
                # Perform step: X_k = Q^-1 * (Q - A) * X_k-1 + Q^-1 * b
                self.Xk = np.matmul(qInvQA, self.Xk) + np.matmul(qInv, self.eqSystem.b)
                # Get residuum Rk of Xk
                Rk = np.matmul(self.eqSystem.A, self.Xk) - self.eqSystem.b
                # Get precision of Rk
                precision: float = linalg.norm(Rk) / linalg.norm(self.eqSystem.b)
                # Return Xk if it satisfies the given precision
                if precision < self.requiredPrecision:
                    print('Step: {}'.format(i))
                    print('Precision: {}'.format(precision))
                    return self.Xk
            # Return None (solution not found) if there was overflow (due to divergence)
            except FloatingPointError:
                print('Solution overflow after {} steps.'.format(i))
                return None
        # Return None (solution on found)
        print('Solution not found after {} steps.'.format(self.maxK))
        return None

    # Get matrix for Jacobi method
    def JacobiMatrix(self):
        self.Q = copy.deepcopy(self.eqSystem.D)

    # Get matrix for GaussSeidel method
    def GaussSeidelMatrix(self):
        self.Q = self.eqSystem.D + self.eqSystem.L

    # Solution by Jacobi method
    def Jacobi(self, eqSystem: InputSystem):
        print('Jacobi')
        self.init(eqSystem)
        self.JacobiMatrix()
        return self.core()

    # Solution by GaussSeidel method
    def GaussSeidel(self, eqSystem: InputSystem):
        print('GaussSeidel')
        self.init(eqSystem)
        self.GaussSeidelMatrix()
        return self.core()
