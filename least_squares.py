import numpy as np
from orthogonal_projection import orthogonal_projection_coefficients as opc
from orthogonal_projection import orthogonal_projection_matrix_LI as opm


def least_squares_coefficients(matrix, target_vector):
    """
        Return least squares coefficients. Note that the input matrix will
        be copied and added a column of 1s vector on the left. If the 
        matrix is not linearly independent, an empty np.ndarray returns.

        paras:
          matrix: np.ndarray
          target_vector: np.ndarray, a column vector contains values to be 
            approximated. This should be n x 1 vector.

        return:
          coefficients: np.ndarray, least squares coefficients.
    """

    matrix = np.array(matrix)
    matrix = np.concatenate((np.ones((matrix.shape[0], 1)), matrix), 1)
    matrix_for_coef = opc(matrix)
    coefficients = np.array([])
    if matrix_for_coef.any():
        coefficients = matrix_for_coef @ target_vector
    return coefficients


def least_squares_approximation(matrix, target_vector):
    """
        Return an approximation vector that approximates a target vector.
        Note that the input matrix will be copied and added a column of 1s 
        vector on the left. If the matrix has any column vector linearly
        dependent of others or all one vector([1...1]), the returned 
        matrix is an empty np.ndarray.

        para:
          matrix: np.ndarray.
          target_vector: np.ndarray, a column vector contains values to be 
            approximated. This should be n x 1 vector.

        return:
          approximation_vector: np.ndarray, the vector that approximates
            target_vector.
    """

    matrix = np.array(matrix)
    matrix = np.concatenate((np.ones((matrix.shape[0], 1)), matrix), 1)
    matrix_least_squares = opm(matrix)
    approximation_vector = np.array([])
    if matrix_least_squares.any():
        approximation_vector = matrix_least_squares @ target_vector
    return approximation_vector


def least_squares_approximation_error(target_vector, approx_vector):
    """
        Return the error of least squares approximation result, which is 
        the sum of the squared difference between target vector and
        approximation vector.

        paras:
          target_vector: np.ndarray, the target vector to be approximated.
          approx_vector: np.ndarray, the approximation vector.

        return:
          error: float, the approximation error.
    """

    error = sum((target_vector - approx_vector) ** 2)[0]
    return error 


if __name__ == '__main__':
    a0 = (np.array([2.6, 2.72, 2.75, 2.67, 2.68])[:, np.newaxis],
          np.array([2.0, 2.1, 2.1, 2.03, 2.04])[:, np.newaxis])
    a1 = (np.ones((5, 1)),
          np.array([2.0, 2.1, 2.1, 2.03, 2.04])[:, np.newaxis])
    
    t = np.array([0, 1, 2, 3, 3.5])[:, np.newaxis]
    a2 = (np.concatenate((t, t**2), 1),
          np.array([100, 118, 92, 48, 7])[:, np.newaxis])

    system = a2
    ls_coef = least_squares_coefficients(*system)
    approx_vector = least_squares_approximation(*system)
    print(f'x:\n{system[0]}')
    print(f'least squares coefficients:\n{ls_coef}')
    if ls_coef.any():
        matrix = np.concatenate((np.ones((system[0].shape[0], 1)), system[0]), axis = 1)
        print(f'least squares approximation:\n{matrix @ ls_coef}')
        print(f'least squares approximation directly:\n{approx_vector}')
        print(f'error\n{least_squares_approximation_error(system[1], approx_vector)}')
