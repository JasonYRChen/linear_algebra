import numpy as np
from inverse_matrix import inverse_matrix


def orthogonal_projection_matrix_LI(matrix):
    """
        Return orthogonal projection matrix of a matrix with linearly
        independent columns. If column vectors of the matrix are not LI,
        an empty np.ndarray returns.

        para:
          matrix: np.ndarray, a matrix with linearly independent columns.

        return:
          op_matrix: np.ndarray, the orthogonal projection matrix.
    """

    inverse_mTm = inverse_matrix(matrix.T @ matrix)
    if not inverse_mTm.any():
        return np.array([])
    op_matrix = matrix @ inverse_mTm @ matrix.T
    return op_matrix


if __name__ == '__main__':
    a0 = np.array([[1, -2], [1, 0], [0, 1]])

    matrix = a0
    print(f'matrix:\n{matrix}')
    print(f'OPM:\n{orthogonal_projection_matrix_LI(matrix)}')
