import numpy as np
from numpy.linalg import norm


def modified_gram_schmidt(matrix):
    """
        Return orthonormal set of column vectors in matrix. Modified Gram-
        Schmidt method is implemented.

        para:
          matrix: np.ndarray, containing column vectors.

        return:
          matrix: np.ndarray, orthonormal column vectors.
    """

    matrix = np.array(matrix)
    cols = matrix.shape[1]
    for i in range(cols):
        matrix[:, i] /= norm(matrix[:, i])
        for j in range(i+1, cols):
            matrix[:, j] -= (matrix[:, j] @ matrix[:, i]) * matrix[:, i]
    return matrix


if __name__ == '__main__':
    a1 = np.array([[1, 2, 1], [1, 1, 1], [1, 0, 2], [1, 1, 1]]).astype(float)

    matrix = a1
    print(f'before:\n{matrix}')
    print(f'MGS:\n{modified_gram_schmidt(matrix)}')
    print(f'after:\n{matrix}')
