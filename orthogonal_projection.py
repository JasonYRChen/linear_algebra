import numpy as np
from inverse_matrix import inverse_matrix


def orthogonal_projection_coefficients(matrix):
    """
        Return a matrix for computing orthogonal projection coefficients.
        If column vectors of input matrix is not linearly independent, an
        empty np.ndarray returns.

        para:
          matrix: np.ndarray, a matrix with linearly independent columns.

        return:
          matrix_for_coef: np.ndarray, matrix for computing coefficients.
    """

    inverse_mTm = inverse_matrix(matrix.T @ matrix)
    if not inverse_mTm.any():
        return np.array([])
    matrix_for_coef = inverse_mTm @ matrix.T
    return matrix_for_coef


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

    matrix_for_coef = orthogonal_projection_coefficients(matrix)
    if not matrix_for_coef.any():
        return matrix_for_coef
    op_matrix = matrix @ matrix_for_coef
    return op_matrix


if __name__ == '__main__':
    a0 = np.array([[1, -2], [1, 0], [0, 1]])
    a1 = np.array([[1, 2], [3, 6]])

    matrix = a1
    print(f'matrix:\n{matrix}')
    print(f'OPM:\n{orthogonal_projection_matrix_LI(matrix)}')
