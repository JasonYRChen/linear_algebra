import numpy as np
from reduced_row_echelon_form import reduced_row_echelon_form
from reduced_row_echelon_form import leading_nonzero
from lu_factorization import lu_factorization


def solve_linear_equations(matrix):
    """
        Solve linear equations with single vector. Returns basic vector
        and free vectors as general solution. If it is an inconsistent
        system, return empty lists for both basic vector and free vectors.
        If the system has one solution, then free vectors is empty. If the
        system has infinite solutions, both vectors are filled.

        The core is the data structure of 'variables', which is a 
        np.ndarray representing a linear combinations of the variable by 
        other variables. The last one element in the array is a constant. 
        Each element in the array refers to the coefficient of the 
        corresponding variable. For example:
        
                             v1 v2 v3  v4 const
        variable v's array = [0, 0, 2, -3, 10], then
        v = 0*v1 + 0*v2 + 2*v3 - 3*v4 + 10

                       v1  v2 v3 const
        If v's array = [0 , 1, 0, 0], it means that v is actually v2 and 
        it's a free variable, since its constant is undetermined.

        para:
          matrix: np.ndarray, the augmented matrix of a linear system.

        returns:
          basic_vector: np.ndarray, the basic part of general solution.
          free_vectors: list of np.ndarray, the free part of general
            solution.
    """

    total_elements = matrix.shape[1] - 1
    variables = np.eye(total_elements, total_elements + 1)
    matrix = reduced_row_echelon_form(matrix, total_elements + 1)

    # solve each variable in terms of variables and constant from top of
    # the matrix to the bottom. If contradiction occur, the system is 
    # inconsistent, then returns None.
    for i, row in enumerate(matrix):
        leading_index = leading_nonzero(matrix, i, total_elements + 1) 
        # contradiction occurs: 0 = 1
        if leading_index == total_elements:
            return [], []
        # 0 = 0 occurs, no need to process further
        if leading_index == -1:
            break

        # update variable with terms other than itself
        variables[leading_index] -= row
        variables[leading_index][-1] *= -1.0

    free_vectors_position = np.any(variables, 0)
    free_vectors_position[-1] = False
    free_vectors = variables[:, free_vectors_position]
    basic_vector = variables[:, -1][:, np.newaxis]

    return basic_vector, free_vectors


def lu_linear_solution(matrix, y):
    """
        Solve a linear system by LU factorization and return both basic
        vector and free vectors in column vectors. If the system is 
        inconsistent, both basic vector and free vectors are empty vectors.

        paras:
          matrix: np.ndarray, the matrix to be LU decomposed.
          y: np.ndarray, m x 1 dimension.
    """

    L, U, row_exchange = lu_factorization(matrix)    
    y = (row_exchange @ np.array(y)).flatten()

    # forward substitution: solve Ly' = y
    for index, row in enumerate(L):
        for i in range(index):
            y[index] -= row[i] * y[i]
        y[index] /= row[index]

    # backward substitution: solve Uz = y' using "solve_linear_equations"
    U = np.concatenate([U, y[:, np.newaxis]], 1)
    basic_vector, free_vector = solve_linear_equations(U)
    return basic_vector, free_vector


if __name__ == '__main__':
    a1 = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0]])
    a2 = np.array([[0, 0, 3], [2, 2, 3]])
    a3 = np.array([[1, -2, -1, 3], [3, -6, -5, 3], [2, -1, 1, 0]])
    a4 = np.array([[0, 0, 7], [0, 0, 0], [3, 0, 3]])
    a5 = np.array([[0, 0, 11, 0], [0, 0, 0, 0], [2, 0, 8, 6], [0, 7, 7, 14]])
    a6 = np.array([[1, -3, 0, 2, 0, 7], [0, 0, 1, 6, 0, 9], [0, 0, 0, 0, 1, 2]])
    a7 = np.zeros((3, 4))
    a8 = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    a9 = np.array([[1, 2, -1, 2, 1, 2], [-1, -2, 1, 2, 3, 6], [2, 4, -3, 2, 0, 3], [-3, -6, 2, 0, 3, 9]])
    a0 = np.array([[1, 0, 0, -3, 0], 
                   [0, 1, 0, -4, 0],
                   [0, 0, 1, 5, 0]])

    matrix = a0
    print(f'original matrix:\n{matrix}')
    s = solve_linear_equations(matrix)
    print('---solutions by "solve_linear_equations"---')
    print(f'basic:\n{s[0]}')
    print(f'free:\n{s[1]}')
    basic, free = lu_linear_solution(matrix[:, :-1], matrix[:, -1])
    print('---solutions by LU---')
    print(f'basic:\n{basic}')
    print(f'free:\n{free}')
