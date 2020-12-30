import numpy as np


class Norm:

    @staticmethod
    def sumNorm(matrix: np.array):
        res: float = 0.0
        for i in range(len(matrix)):
            row = matrix[i]
            for j in range(len(row)):
                res += row[j]
        return res
