import numpy as np
from reduced_row_echelon_form import reduced_row_echelon_form


def rank(matrix):
    """
        Calculate rank of a matrix.

        para:
          matrix: matrix-like object which can be np.array(matrix).

        return:
          rank: int, rank of the matrix
    """

    matrix = reduced_row_echelon_form(matrix)
    rank = 0
    for row in matrix:
        if not any(row):
            break
        rank += 1
    return rank


def nullity(matrix):
    """
        Calculate nullity of a matrix.

        para:
          matrix: matrix-like object which can be np.array(matrix).

        return:
          nullity: int, nullity of the matrix.
    """

    rank_= rank(matrix)
    nullity = matrix.shape[1] - rank_
    return nullity


if __name__ == '__main__':
    a0 = np.array([[2, 3, 1, 5, 2],
                   [0, 1, 1, 3, 2], 
                   [4, 5, 1, 7, 2],
                   [2, 1, -1, -1, -2]
                  ])
    matrix = a0
    print(f'rank: {rank(matrix)}')
    print(f'nullity: {nullity(matrix)}')
