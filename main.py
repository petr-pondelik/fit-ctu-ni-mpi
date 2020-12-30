import numpy as np
from numpy import linalg

from Algorithm.Algorithm import Algorithm
from Matrix.InputSystem import InputSystem

# Test
m = np.array([[2, 3], [4, 5]])
mInv = linalg.inv(m)
# print(np.matmul(m, mInv))

eqSystem: InputSystem = InputSystem()
algorithm: Algorithm = Algorithm()
algorithm.Jacobi(eqSystem)
