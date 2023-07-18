import numpy as np
from elementary_row_operation import *


# scaling(matrix, row_index, scalar)
# row_addition(matrix, from_row, scalar, to_row)
# interchange(matrix, row1, row2)

"""
    Beware that row echelon needs non-zero leading element at the front
    of each column. You may need to check the first non-zero element in
    each column.

    case1.
      [[0, 0, 1],
       [1, 2, 3]]
      --> [[1, 2, 3],
           [0, 0, 1]]
    case2.
      [[0, 0, 1],
       [0, 1, 1]]
      --> [[0, 1, 1],
           [0, 0, 1]]
    case3.
      [[0, 0, 1],
       [0, 0, 0],
       [1, 0, 3]]
      --> [[1, 0, 3],
           [0, 0, 1],
           [0, 0, 0]]
"""


def pivot_column(matrix, pivot):
    """
        Find the minimum of possible pivot column. The actual column can 
        only be larger than or equal to the result.

        paras:
          matrix: np.ndarray
          pivot: int, the pivot point to find.

        return:
          column: int, the minimum of possible pivot column.
    """

    if pivot:
        column = (matrix[pivot - 1] != 0).argmax() + 1
    else:
        column = 0
    return column


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
        pivot position. This is an in-place operation.

        paras:
          matrix: np.ndarray.
          pivot: int, the pivot position to find.
          column: int, the first column to start to find.

        return:
          matrix: np.ndarray.
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

    return matrix


def reduced_row_echelon_form(matrix):
    """

    """

    matrix = np.array(matrix)
    pivot = 0
    column = 0
    for _ in range(min(matrix.shape) - 1):
        matrix = set_pivot_row(matrix, pivot, column)
        column = leading_nonzero(matrix, pivot)
        if (column == matrix.shape[1] - 1) or (column == -1):
            break
        pivot += 1
        column += 1

    return matrix


if __name__ == '__main__':
    from elementary_row_operation import to_ndarray
    a1 = to_ndarray([[r*3+c for c in range(1, 4)] for r in range(0, 4)])
    a2 = np.array([[0, 0, 1], [1, 2, 3]])
    a3 = np.array([[0, 0, 1], [0, 1, 1]])
    a4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 3]])
    a5 = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 1], [0, 1, 1, 1]])
    a6 = np.array([[0, 1, 0], [1, 0, 0]])
    a7 = np.zeros((3, 3))
    a8 = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])

    matrix = a7
    print(matrix)
    print(reduced_row_echelon_form(matrix))
