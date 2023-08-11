import numpy as np
from inverse_matrix import inverse_matrix
from qr_method import qr_method


def diagonalization(matrix):
    """
        Diagonalize a square matrix and return its eigenvectors matrix, 
        diagonalized  matrix, and inverse of eigenvectors matrix in a 
        tuple. The diagonalization method is borrowed from numpy.linalg.eig.

        para:
          matrix: np.ndarray.

        returns (in a tuple):
          e_vectors: np.ndarray, a matrix of eigenvectors of input matrix.
          diagonal: np.ndarray, diagonal matrix with its diagonal entries
            consist of eigenvalues.
          inverse_e_vectors: np.ndarray, an inverse matrix of e_vectors.
    """

    #e_values, e_vectors = np.linalg.eig(matrix) # use built-in function
    e_values, e_vectors = qr_method(matrix) # use self-made function
    diagonal = np.eye(len(e_values))
    for i, value in enumerate(e_values):
        diagonal[:, i] *= value
    inverse_e_vectors = inverse_matrix(e_vectors)
    return e_vectors, diagonal, inverse_e_vectors


if __name__ == '__main__':
    a0 = np.array([[-4, -3], [3, 6]])
    a1 = np.array([[-1, 0, 0], [0, 1, 2], [0, 2, 1]])

    matrix = a1
    p, d, p_inv = diagonalization(matrix)
    print(matrix)
    print(f'{p}\n{d}')
    print(p @ d @ p_inv)
