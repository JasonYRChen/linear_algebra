import numpy as np
from inverse_matrix import inverse_matrix

"""
This module is for computing orthogonal projection, or least squares of, a
vector on a system. The system itself must be linearly independent, or the
floating error may cause unexpected false.
"""

def orthogonal_projection_coefficient(matrix, y, need_intercept=False):
    """
        Return the coeffieicent of projection vector in the matrix of y.
        If "need_intercept" is True, then an additional 1s vector ([1...1])
        will be concatenated on the left of the matrix as an intercept. If
        the matrix is not column-wise linearly independent, an empty
        np.array returns, or floating error occurs and crashes the result
        without warnings.

        paras:
          matrix: np.ndarray, the system.
          y: np.ndarray, an nx1 array to be fitted.
          need_intercept: bool, need an extra column for intercept or not.
            If True, an extra 1s column will be concatenated on the left of
            the matrix.

        return:
          coefficient: np.ndarray, an nx1 array of coefficient.
    """

    if need_intercept:
        matrix = np.array(matrix)
        matrix = np.concatenate((np.ones((matrix.shape[0], 1)), matrix), 1)
    inverse_mTm = inverse_matrix(matrix.T @ matrix)
    coefficient = np.array([])
    if inverse_mTm.any():
        coefficient = inverse_mTm @ matrix.T @ y
    return coefficient


def orthogonal_projection_approximation(matrix, y, need_intercept=False):
    """
        Return the approximation of projection vector in the matrix of y.
        If "need_intercept" is True, then an additional 1s vector ([1...1])
        will be concatenated on the left of the matrix as an intercept. If
        the matrix is not column-wise linearly independent, an empty
        np.array returns, or floating error occurs and crashes the result
        without warnings.

        paras:
          matrix: np.ndarray, the system.
          y: np.ndarray, an nx1 array to be fitted.
          need_intercept: bool, need an extra column for intercept or not.
            If True, an extra 1s column will be concatenated on the left of
            the matrix.

        return:
          approximation_vector: np.ndarray, an nx1 array of approximation.

    """

    if need_intercept:
        matrix = np.array(matrix)
        matrix = np.concatenate((np.ones((matrix.shape[0], 1)), matrix), 1)
    coefficient = orthogonal_projection_coefficients(matrix, y)
    approximation_vector = np.array([])
    if coefficient.any():
        approximation_vector = matrix @ coefficient
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
