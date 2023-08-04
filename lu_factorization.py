import numpy as np
from elementary_row_operation import *
from reduced_row_echelon_form import pivot_row
from inverse_matrix import inverse_matrix


def row_elimination(matrix, pivot_row, pivot_col):
    """
        Return a Gaussian eliminated matrix and its corresponding 
        elementary operation matrix along a specified column. Only rows 
        below 'pivot_row' are eliminated.

        paras:
          matrix: np.ndarray
          pivot_row: int, along with 'pivot_col' to locate pivot position.
          pivot_col: int.

        returns:
          matrix: np.ndarray, the row eliminated matrix.
          general_elem_matrix: np.ndarray, the product of elementary
            matrices produced during row addition processes.
    """

    pivot_scalar = matrix[pivot_row, pivot_col]
    general_elem_matrix = np.eye(matrix.shape[0])
    for r in range(pivot_row+1, matrix.shape[0]):
        scalar = -matrix[r, pivot_col] / pivot_scalar
        matrix = row_addition(matrix, pivot_row, scalar, r)
        elem_matrix = row_addition(np.eye(matrix.shape[0]), pivot_row, 
                                   scalar, r)
        general_elem_matrix = elem_matrix @ general_elem_matrix
    return matrix, general_elem_matrix


def permutation_for_lu(matrix):
    """
        Return the necessary row permutation matrix of matrix before doing
        LU factorization. It also returns L and U and can be used 
        immediately if the row permutation matrix is an identity.

        para:
          matrix: np.ndarray.

        returns:
          L: np.ndarray, lower triangular matrix.
          U: np.ndarray, upper triangular matrix.
          E: np.ndarray, row permutation matrix.
    """

    U = np.array(matrix).astype(float)
    inv_L = np.eye(U.shape[0])
    row_exchange = np.eye(U.shape[0])
    for row in range(U.shape[0]):
        # check pivot, if row exchange happen, multiply the interchange 
        # result matrix to 'row_exchange'
        pivot_r, pivot_c = pivot_row(U, row, U.shape[1])
        if pivot_r == -1: # rests are 0 row vectors, end factorizing
            break
        if pivot_r != row:
            U = interchange(U, pivot_r, row)
            exchange = interchange(np.eye(U.shape[0]), pivot_r, row)
            row_exchange = exchange @ row_exchange

        # row elimination other rows
        U, elem_matrix = row_elimination(U, row, pivot_c)
        inv_L = elem_matrix @ inv_L
    L = inverse_matrix(inv_L)

    return L, U, row_exchange


def lu_factorization(matrix):
    """
        Return matrix's LU factorization. If the matrix needs row 
        permutation to have LU factorization, then the resulting L and U
        are computed as the matrix is being permutated.

        para:
          matrix: np.ndarray.

        returns:
          L: np.ndarray, lower triangular matrix.
          U: np.ndarray, upper triangular matrix.
          E: np.ndarray, row permutation matrix.
    """

    L, U, row_exchange = permutation_for_lu(matrix)
    if not np.array_equal(row_exchange, np.eye(matrix.shape[0]).astype(float)):
        U = row_exchange @ np.array(matrix).astype(float)
        inv_L = np.eye(U.shape[0])
        for row in range(U.shape[0]):
            pivot_r, pivot_c = pivot_row(U, row, U.shape[1])
            if pivot_r == -1:
                break

            U, elem_matrix = row_elimination(U, row, pivot_c)
            inv_L = elem_matrix @ inv_L
        L = inverse_matrix(inv_L)

    return L, U, row_exchange


if __name__ == '__main__':
    a2 = np.array([[0, 0, 3], [2, 2, 3]])
    a3 = np.array([[0, 0, 1], [0, 3, 9]])
    a4 = np.array([[0, 0, 7], [0, 0, 0], [3, 0, 3]])
    a5 = np.array([[0, 0, 11, 0], [0, 0, 0, 0], [2, 0, 8, 6], [0, 7, 7, 14]])
    a6 = np.array([[0, 9, 0], [5, 0, 0]])
    a7 = np.zeros((3, 3))
    a8 = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    a9 = np.array([[1, 2, -1, 2, 1, 2], [-1, -2, 1, 2, 3, 6], [2, 4, -3, 2, 0, 3], [-3, -6, 2, 0, 3, 9]])
    exchange9 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    a0 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a1 = np.array([[0, 3, 4, 5], [0, 6, 9, 3], [0, 0, 0, 7], [2, 3, 7, 5]])
    a10 = np.array([[0, 2, 2, 4], [0, 2, 2, 2], [1, 2, 2, 1], [2, 6, 7, 5]])

    matrix = a10
    L, U, E = lu_factorization(matrix)
    result = permutation_for_lu(matrix)
    print('----result----')
    print(f'matrix:\n{matrix}')
    print(f'L:\n{L}')
    print(f'U:\n{U}')
    print(f'exchange:\n{E}')
    print(f'exchanged matrix:\n{E @ matrix}')
    print(f'L @ U:\n{L @ U}')
