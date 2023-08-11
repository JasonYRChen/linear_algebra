import numpy as np
from gram_schmidt import modified_gram_schmidt
from solve_equation import solve_linear_equations


def qr_decomposition(matrix):
    """
        QR decompose a matrix with linearly independent Col(matrix) and 
        return Q and R matrices. Modified Gram-Schmidt method is used to
        calculate orthonormal column vectors Q.

        para:
          matrix: np.ndarray, m x n matrix that Col(matrix) must be 
            linearly independent.

        return:
          Q: np.ndarray, m x n matrix, orthonormal column vectors from
            matrix.
          R: np.ndarray, n x n and upper triangular matrix.
    """

    Q = modified_gram_schmidt(matrix)
    R = Q.T @ matrix
    R -= np.tril(R, -1)
    return Q, R


def _is_upper_triangle(matrix, accuracy):
    """
        Return True if the matrix is upper triangular. Since floating error
        may occur during computation, a float 'accuracy' is used to identify
        zero if a number is smaller than 'accuracy'.

        paras:
          matrix: np.ndarray, m x n matrix.
          accuracy: float, number smaller than this is identified zero.

        return:
          is_upper: bool, True if the matrix is upper triangular or False 
            otherwise.
    """

    is_upper = (np.tril(matrix, -1) <= accuracy).all()
    return is_upper


def qr_method(matrix, decimal=7):
    """
        Return eigenvalues and eigenvectors of a matrix. QR iteration
        method is used under the hood.

        paras:
          matrix: np.ndarray, n x n matrix.
          decimal: int, specified digit of decimal.

        returns:
          eigenvalues: list, a list of eigenvalues of the matrix.
          eigenvectors: np.ndarray, each column represents an eigenvetor of
            the corresponding eigenvalue of the same index.
    """

    mat = np.array(matrix)
    accuracy = 1 / 10 ** (decimal + 1) # '+1' here to avoid rounding error
                                       # at rounding 'e_values'
    while not _is_upper_triangle(mat, accuracy):
        Q, R = qr_decomposition(mat)
        mat = R @ Q

    e_values = np.array(np.diag(mat)) # np.diag could only return the
                                      # view of diagonal
    e_values = np.round(e_values, decimal)
    eigenvalues = []
    eigenvectors = None
    rows = matrix.shape[0]
    identity = np.eye(rows)
    used_eigenvalues = set()
    for eigenvalue in e_values:
        if eigenvalue not in used_eigenvalues:
            used_eigenvalues.add(eigenvalue)
            mat = matrix - eigenvalue * identity
            mat = np.hstack((mat, np.zeros((rows, 1))))
            _, free_vectors = solve_linear_equations(mat)
            eigenvalues.extend([eigenvalue] * free_vectors.shape[1])

            # normalize eigenvectors
            for i in range(free_vectors.shape[1]):
                free_vectors[:, i] /= ((free_vectors[:, i] ** 2).sum()) ** 0.5
            if eigenvectors is None:
                eigenvectors = free_vectors
            else:
                eigenvectors = np.hstack((eigenvectors, free_vectors))
    eigenvalues = np.array(eigenvalues)

    return eigenvalues, eigenvectors


if __name__ == '__main__':
    m1 = np.array([[1, 1, 0], [1, -1, 0], [1, 1, 1], [1, 1, 1]])
    m2 = np.array([[9, 8], [1, 2]])
    m3 = np.array([[1, 1], [1, 0]])

    matrix = m2
    eigenvalues, eigenvectors = qr_method(matrix)
    print(f'matrix:\n{matrix}')
    print(f'eigenvalues:\n{eigenvalues}')
    print(f'eigenvectors:\n{eigenvectors}')
