import numpy as np
from elementary_row_operation import *


def leading_nonzero(matrix, row):
    """
        Return the index of leading non-zero element in the row.

        paras:
          matrix: np.ndarray.
          row: int, the row to find the index of leading non-zero element.

        return:
          index: int, index of the leading non-zero element.
    """

    indices = np.where(matrix[row] != 0)[0]
    index = indices[0] if indices.size else -1
    return index


def set_pivot_row(matrix, pivot, column):
    """
        Interchange two rows in the matrix to set pivot row at current
        pivot position. Return the interchanged matrix and next pivot and
        next column to deal with.

        paras:
          matrix: np.ndarray.
          pivot: int, the pivot position to find.
          column: int, the first column to start to find.

        return:
          matrix: np.ndarray.
          next_pivot: int, the next pivot.
          next_column: int, the next column.
    """

    # column = leading_nonzero(matrix, pivot - 1) + 1 if pivot else 0
    for c in range(column, matrix.shape[1]):
        for r in range(pivot, matrix.shape[0]):
            if matrix[r][c]:
                matrix = interchange(matrix, pivot, r)
                break
        else:
            continue
        break

    # determine the next pivot and column
    column = leading_nonzero(matrix, pivot)
    if (column == matrix.shape[1] - 1) or (column == -1):
        column = -2
    next_pivot = pivot + 1
    next_column = column + 1
    return matrix, next_pivot, next_column


def reduced_row_echelon_form(matrix):
    """
        Calculate reduced row echelon form of input matrix.

        para:
          matrix: np.ndarray

        return:
          matrix: np.ndarray, the reduced row echelon form of input matrix.
    """

    matrix = np.array(matrix) # copy original matrix
    pivot = 0
    column = 0

    for _ in range(min(matrix.shape)):
        # find and set pivot row
        matrix, pivot, column = set_pivot_row(matrix, pivot, column)

        # scaling the pivot row by inverse of leading nonzero number 
        pivot_column = leading_nonzero(matrix, pivot - 1)
        leading_number = matrix[pivot - 1][pivot_column]
        scalar = 1 / leading_number if leading_number else 0
        matrix = scaling(matrix, pivot - 1, scalar)

        # row addition of pivot row and other rows
        for row in range(matrix.shape[0]):
            if row != pivot - 1:
                scalar = -matrix[row][pivot_column]
                matrix[row] += matrix[pivot - 1] * scalar

        # remaining rows are zeros, no need to calculate
        if column == -1:
            break

    return matrix


if __name__ == '__main__':
    from elementary_row_operation import to_ndarray
    a1 = to_ndarray([[r*3+c for c in range(1, 4)] for r in range(0, 4)])
    a2 = np.array([[0, 0, 3], [2, 2, 3]])
    a3 = np.array([[0, 0, 1], [0, 3, 9]])
    a4 = np.array([[0, 0, 7], [0, 0, 0], [3, 0, 3]])
    a5 = np.array([[0, 0, 11, 0], [0, 0, 0, 0], [2, 0, 8, 6], [0, 7, 7, 14]])
    a6 = np.array([[0, 9, 0], [5, 0, 0]])
    a7 = np.zeros((3, 3))
    a8 = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    a9 = np.array([[1, 2, -1, 2, 1, 2], [-1, -2, 1, 2, 3, 6], [2, 4, -3, 2, 0, 3], [-3, -6, 2, 0, 3, 9]])

    matrix = a9
    print(matrix)
    print(reduced_row_echelon_form(matrix))
