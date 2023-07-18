import numpy as np


__all__ = [
           'interchange',
           'scaling',
           'row_addition'
          ]


def to_ndarray(array):
    """
        Convert non-ndarray but array-like array into np.ndarray. If array
        is np.ndarray, return immediately. The return ndarray is forced to
        cast to float type.

        para:
          array: List-like

        return:
          array: np.ndarray, return a float type matrix.
    """

    if not isinstance(array, np.ndarray):
        array = np.array(array)
    array = array.astype(float)
    return array


def interchange(matrix, row1, row2):
    """
        Return a np.ndarray with designated two rows interchanged.

        paras:
          matrix: List-like, representing a matrix.
          row1 & row2: int, specify the two rows to interchange.

        return:
          matrix: np.ndarray, the interchanged matrix.
    """

    matrix = to_ndarray(matrix)
    temp = matrix[row1].copy()
    matrix[row1], matrix[row2] = matrix[row2], temp
    return matrix


def scaling(matrix, row_index, scalar):
    """
        Multiply a scalar to specified row.

        paras:
          matrix: List-like, representing a matrix.
          row_index: int, specifies the row to multiply.
          scalar: int/float, the scalar to multiply.

        return:
          matrix: np.ndarray.
    """

    matrix = to_ndarray(matrix)
    matrix[row_index] *= scalar
    return matrix


def row_addition(matrix, from_row, scalar, to_row):
    """
        Add a scaled row to another row in matrix.

        paras:
          matrix: List-like, representing a matrix.
          from_row: int, the row to be scaled and added to another row.
          scalar: int/float, the scalar to multiply to "from_row" row.
          to_row: int, the row to be added.

       return:
          matrix: np.ndarray.
    """

    matrix = to_ndarray(matrix)
    matrix[to_row] += matrix[from_row] * scalar
    return matrix   


if __name__ == '__main__':
    array = [[r*3+c for c in range(1, 4)] for r in range(0, 4)]
    # array = np.array(array)
    print(array)
    print(row_addition(array, 1, 4.0, 2))
