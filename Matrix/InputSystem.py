import copy
from typing import List
import numpy as np


class InputSystem:

    n: int
    gamma: float

    A: np.array
    b: np.array
    L: np.array
    D: np.array
    U: np.array

    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.A = np.array(self.constructA())
        self.b = np.array(self.constructB())
        self.L = self.extractL()
        self.D = self.extractD()
        self.U = self.extractU()

    # Construct equations system matrix
    def constructA(self) -> List[List[float]]:
        # Prepare empty system matrix
        A: List[List[float]] = [[] for i in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                # Fill matrix with zeroes
                A[i].append(0.0)
                # Set diagonal items to gamma
                if i == j:
                    A[i][j] = self.gamma
                # Set items around diagonal to -1
                if abs(i - j) == 1:
                    A[i][j] = -1.0
        return A

    # Construct equations system right side vector
    def constructB(self) -> List[List[float]]:
        b: List[List[float]] = [[] for i in range(self.n)]
        for i in range(self.n):
            if i == 0 or i == (self.n-1):
                b[i].append(self.gamma - 1)
            else:
                b[i].append(self.gamma - 2)
        return b

    # Extract L part from A matrix
    def extractL(self) -> np.array:
        L: np.array = copy.deepcopy(self.A)
        for i in range(self.n):
            for j in range(self.n):
                if i <= j:
                    L[i][j] = 0
        return L

    # Extract D part from A matrix
    def extractD(self) -> np.array:
        D: np.array = copy.deepcopy(self.A)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    D[i][j] = 0
        return D

    def extractU(self) -> np.array:
        return self.A - self.L - self.D
