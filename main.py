from Algorithm.Algorithm import Algorithm
from Matrix.InputSystem import InputSystem

# Construct equations systems to be solved
eqSystem1: InputSystem = InputSystem(20, 3.0)
eqSystem2: InputSystem = InputSystem(20, 2.0)
eqSystem3: InputSystem = InputSystem(20, 1.0)

# Construct algorithm with the given max steps and required precision
algorithm: Algorithm = Algorithm(1000, 0.000001)

# Solve equations using Jacobi algorithm
algorithm.Jacobi(eqSystem1)
algorithm.Jacobi(eqSystem2)
algorithm.Jacobi(eqSystem3)