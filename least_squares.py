import numpy as np
from orthogonal_projection import orthogonal_projection_coefficients as opc
from orthogonal_projection import orthogonal_projection_matrix_LI as opm


def least_squares_coefficients(matrix):
    """
        Return a matrix for computing least squares coefficients. If the
        input matrix is not linearly independent, an empty np.ndarray 
        returns.

        para:
          matrix: np.ndarray, a matrix with linearly independent columns.

        return:
          matrix_for_coef: np.ndarray, matrix for computing coefficients.
    """

    matrix = np.array(matrix)
    matrix = np.concatenate((np.ones((matrix.shape[0], 1)), matrix), 1)
    matrix_for_coef = opc(matrix)

    return matrix_for_coef


def least_squares_matrix(matrix):
    """
        Return a matrix to compute least squares approximation. If the
        input matrix has any column vector linearly dependent of others
        or all one vector([1...1]), the returned matrix is an empty
        np.ndarray.

        para:
          matrix: np.ndarray.

        return:
          matrix_least_squares: np.ndarray, matrix for computing least
            squares approximation.
    """

    matrix = np.array(matrix)
    matrix = np.concatenate((np.ones((matrix.shape[0], 1)), matrix), 1)
    matrix_least_squares = opm(matrix)
    return matrix_least_squares


if __name__ == '__main__':
    a0 = (np.array([2.6, 2.72, 2.75, 2.67, 2.68])[:, np.newaxis],
          np.array([2.0, 2.1, 2.1, 2.03, 2.04])[:, np.newaxis])

    system = a0
    ls_coef = least_squares_coefficients(system[0])
    print(f'x:\n{system[0]}')
    print(f'least squares coefficients:\n{ls_coef}')
    if ls_coef.any():
        matrix = np.concatenate((np.ones((system[0].shape[0], 1)), system[0]), axis = 1)
        print(f'least squares approximation:\n{ls_coef @ system[1]}')
        print(f'least squares approximation:\n{matrix @ ls_coef @ system[1]}')
        print(f'least squares approximation directly:\n{least_squares_matrix(system[0]) @ system[1]}')
