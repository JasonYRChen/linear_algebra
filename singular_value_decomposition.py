import numpy as np
from rank_and_nullity import rank
from solve_equation import solve_linear_equations
from gram_schmidt import modified_gram_schmidt


def singular_value_decomposition(matrix):
    """
        Return the singular value decomposition of the matrix. Three
        matrices u_matrix, sigma_matrix and v_matrix compose of the matrix
        by u_matrix @ sigma_matrix @ transpose(v_matrix).

        para:
          matrix: np.ndarray, the matrix to be decomposed.

        returns:
          u_matrix: np.ndarray, an orthonormal base.
          s_matrix: np.ndarray, singular values are on the diagonal of the 
            matrix, the rest elements are zeros.
          v_matrix: np.ndarray, another orthonormal base.
    """

    mat = np.array(matrix)
    rank_ = rank(matrix)
    # do minimum calculation for finding eigen-solution
    if mat.shape[0] < mat.shape[1]: 
        mat = mat.T

    # find one of the basis spanning its dimension of the matrix
    eig_values, eig_vectors1 = np.linalg.eig(mat.T @ mat)
    # sort eig_values and eig_vectors according to descending eig_values
    sequence = np.argsort(eig_values)[::-1]
    eig_values = eig_values[sequence][:rank_]
    eig_vectors1 = eig_vectors1[:, sequence]

    # find the Sigma matrix
    sigmas = [v**0.5 for v in eig_values] # singular values
    sigma_matrix = np.zeros_like(matrix).astype(float)
    for i, sigma in enumerate(sigmas):
        sigma_matrix[i, i] = sigma

    # find the other basis spanning the other dimension of the matrix
    eig_vectors2 = (mat @ eig_vectors1[:, :rank_]) / sigmas
    # if eig_vectors2 do not fully span its space, the following execute
    if rank_ < mat.shape[0]:
        # the rest vectors to span its space with eig_vectors2 falls in the
        # null space of mat.T
        mat = np.concatenate((mat.T, np.zeros(mat.shape[1])[:, np.newaxis]), 1)
        _, null_vectors = solve_linear_equations(mat)
        ortho_null_vectors = modified_gram_schmidt(null_vectors) # normalize
        eig_vectors2 = np.concatenate((eig_vectors2, ortho_null_vectors), 1)

    if matrix.shape[0] < matrix.shape[1]:
        u_matrix, v_matrix = eig_vectors1, eig_vectors2
    else:
        u_matrix, v_matrix = eig_vectors2, eig_vectors1

    return u_matrix, sigma_matrix, v_matrix
