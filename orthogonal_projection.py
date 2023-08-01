import numpy as np
from inverse_matrix import inverse_matrix
from rank_and_nullity import rank
from singular_value_decomposition import singular_value_decomposition


"""
This module offers orthogonal projection solution to both linearly
dependent and independent systems. The functions with "nonLI" at the end of
their names are actually general solutions for both systems, but they use
eigenform for finding solutions, which may introduce extra floating errors.
It is highly recommended to use "nonLI" functions when the system is 
linearly dependent.
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


def orthogonal_projection_coefficient_nonLI(matrix, 
                                            y, need_intercept=False):
    """
        Return the coeffieicent of projection vector in the matrix of y.
        If "need_intercept" is True, then an additional 1s vector ([1...1])
        will be concatenated on the left of the matrix as an intercept.
        This is the general solution for both linearly dependent and
        independent system. But, since eigen calculation is used, extra
        floating errors could be introduced into the result, it is highly
        recommended to use it if the system is linearly dependent.

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
    U, S, V = singular_value_decomposition(matrix)
    S_dagger = S.T
    for i in range(min(S_dagger.shape[0], S_dagger.shape[1])):
        if not S_dagger[i, i]:
            break
        S_dagger[i, i] = 1 / S_dagger[i, i]
    coefficient = V @ S_dagger @ U.T @ y
    return coefficient


def orthogonal_projection_approximation_nonLI(matrix, 
                                              y, need_intercept=False):
    """
        Return the approximation of projection vector in the matrix of y.
        If "need_intercept" is True, then an additional 1s vector ([1...1])
        will be concatenated on the left of the matrix as an intercept. 
        This is the general solution for both linearly dependent and
        independent system. But, since eigen calculation is used, extra
        floating errors could be introduced into the result, it is highly
        recommended to use it if the system is linearly dependent.

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
    U, S, _ = singular_value_decomposition(matrix)
    rank_ = rank(S)
    D = np.zeros_like(U)
    for i in range(rank_):
        D[i, i] = 1
    approximation = U @ D @ U.T @ y
    return approximation


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
    s0 = (np.array([[0, 1, 2], [1, 0, 1]]),
          np.array([5, 1])[:, np.newaxis])
    s1 = (np.array([[1, 1, 2], [1, -1, 3], [1, 3, 1]]), 
          np.array([1, 4, -1])[:, np.newaxis])
    t = np.array([0, 1, 2, 3, 3.5])[:, np.newaxis]
    s2 = (np.concatenate((t, t**2), 1),
          np.array([100, 118, 92, 48, 7])[:, np.newaxis])

    system = s1
    U, S, V = singular_value_decomposition(system[0])
    coef = orthogonal_projection_coefficient_nonLI(*system)
    approx = orthogonal_projection_approximation_nonLI(*system)
    coef_1 = orthogonal_projection_coefficient_nonLI(*system, True)
    approx_1 = orthogonal_projection_approximation_nonLI(*system, True)
    print(f'U:\n{U}')
    print(f'S:\n{S}')
    print(f'V:\n{V}')
    print(f'coefficient:\n{coef}')
    print(f'approximation:\n{approx}')
    print(f'coefficient with intercept:\n{coef_1}')
    print(f'approximation with intercept:\n{approx_1}')
