import numpy as np
from elementary_row_operation import *


def leading_nonzero(matrix, row, max_column_number):
    """
        Return the index of leading non-zero element in the row.

        paras:
          matrix: np.ndarray.
          row: int, the row to find the index of leading non-zero element.
          max_column_number: int, the maximum number of column to find
            leading nonzero element.

        return:
          index: int, index of the leading non-zero element.
    """

    indices = np.where(matrix[row][:max_column_number] != 0)[0]
    index = indices[0] if indices.size else -1
    return index


def set_pivot_row(matrix, pivot, column, max_column_number):
    """
        Interchange two rows in the matrix to set pivot row at current
        pivot position. Return the interchanged matrix and next pivot and
        next column to deal with.

        paras:
          matrix: np.ndarray.
          pivot: int, the pivot position to find.
          column: int, the first column to start to find.
          max_column_number: int, the maximum number of column to find
            leading nonzero element.

        return:
          matrix: np.ndarray.
          next_pivot: int, the next pivot.
          next_column: int, the next column.
    """

    max_col = min(matrix.shape[1], max_column_number)
    for c in range(column, max_col):
        for r in range(pivot, matrix.shape[0]):
            if matrix[r][c]:
                matrix = interchange(matrix, pivot, r)
                break
        else:
            continue
        break

    # determine the next pivot and column
    column = leading_nonzero(matrix, pivot, max_column_number)
    if (column >= max_col - 1) or (column == -1):
        column = -2
    next_pivot = pivot + 1
    next_column = column + 1
    return matrix, next_pivot, next_column


def reduced_row_echelon_form(matrix, max_column_number=0):
    """
        Calculate reduced row echelon form of input matrix.

        para:
          matrix: np.ndarray
          max_column_number: int, the maximum number of column to find
            leading nonzero element. Default zero means to use the total
            matrix column numbers.

        return:
          matrix: np.ndarray, the reduced row echelon form of input matrix.
    """

    matrix = np.array(matrix).astype(float) # copy original matrix
    pivot = 0
    column = 0
    max_col = max_column_number if max_column_number else matrix.shape[1]

    for _ in range(min(matrix.shape)):
        # find and set pivot row
        matrix, pivot, column = set_pivot_row(matrix, pivot, column, max_col)
        # scaling the pivot row by inverse of leading nonzero number 
        pivot_column = leading_nonzero(matrix, pivot - 1, max_col)
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


def pivot_position(matrix, max_column_number=0):
    """
        Return the pivot column index of a reduced row echelon matrix.

        para:
          matrix: np.ndarray, it must be in reduced row echelon form.
          max_column_number: int, specify the max column number to search
            for pivot column. If it is 0, the whole range of columns will
            be searched.

        return:
          column_index: list, the indices of pivot columns.
    """

    max_col = max_column_number if max_column_number else matrix.shape[1]
    row, col = 0, 0
    column_index = []
    while col != -1 and row < matrix.shape[0]:
        col = leading_nonzero(matrix, row, max_col)
        if col != -1:
            column_index.append(col)
            row += 1
    return column_index


def column_vectors(matrix):
    """
        Return column vectors of a matrix. Those column vectors form a 
        basis for the matrix.

        para:
          matrix: np.ndarray.

        return:
          columns: np.ndarray of ndarray, the column vectors.
    """

    reduced_matrix = reduced_row_echelon_form(matrix)
    indices = pivot_position(reduced_matrix)
    columns = matrix[:, indices]
    return columns


if __name__ == '__main__':
    a2 = np.array([[0, 0, 3], [2, 2, 3]])
    a3 = np.array([[0, 0, 1], [0, 3, 9]])
    a4 = np.array([[0, 0, 7], [0, 0, 0], [3, 0, 3]])
    a5 = np.array([[0, 0, 11, 0], [0, 0, 0, 0], [2, 0, 8, 6], [0, 7, 7, 14]])
    a6 = np.array([[0, 9, 0], [5, 0, 0]])
    a7 = np.zeros((3, 3))
    a8 = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    a9 = np.array([[1, 2, -1, 2, 1, 2], [-1, -2, 1, 2, 3, 6], [2, 4, -3, 2, 0, 3], [-3, -6, 2, 0, 3, 9]])
    a0 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a0 = np.concatenate((a0, np.identity(3)), axis=1)

    matrix = a0
    max_col = 3
    print(matrix)
    print(reduced_row_echelon_form(matrix, max_col))
    print(column_vectors(matrix))
