import numpy as np
from reduced_row_echelon_form import reduced_row_echelon_form


def inverse_matrix(matrix):
    """
        Calculate inverse matrix and return using Gaussian elimination.
        If the matrix is not inversible, return empty np.ndarray instead.

        para:
          matrix: np.ndarray, the matrix to inverse.

        return:
          inverse_matrix: np.ndarray
    """

    matrix = np.array(matrix) # copy original matrix
    dim_row, dim_col = matrix.shape[0], matrix.shape[1]
    inverse_matrix = np.array([])

    if dim_row == dim_col:
        # Using Gaussian elimination to find inverse matrix
        identity = np.identity(dim_row)
        matrix = np.concatenate((matrix, identity), axis=1)
        matrix = reduced_row_echelon_form(matrix, dim_row)

        reduced_matrix = matrix[:, :dim_row] # leading nonzero must be 1
        nonzero_row, nonzero_col = np.where(reduced_matrix != 0)
        # check if reduced matrix is an identity matrix
        if len(nonzero_row) == dim_row:
            for row, col in zip(nonzero_row, nonzero_col):
                if row != col:
                    break
            else:
                inverse_matrix = matrix[:, dim_row:]
    return inverse_matrix


if __name__ == '__main__':
    a0 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a1 = np.array([[1, 2, 3], [4, 5, 6]])
    a2 = np.identity(4)
    a3 = np.array([[1, 2, 3], [2, 5, 6], [3, 4, 8]])
    a4 = np.array([[1, 1, 2], [2, 1, 1], [1, 0, -1]])
    a5 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    a6 = np.array([[1, 2, 1], [2, 5, 1], [2, 4, 1]])
    a9 = np.array([[1, 1, 2], [2, -1, 1],[2, 3, 4]])
    
    matrix = a9
    inversed_matrix = inverse_matrix(matrix)
    print(inversed_matrix)
    if inversed_matrix.size != 0:
        print(matrix @ inversed_matrix)
