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


def _matrix_preprocess_for_iteration(matrix, constants):
    """
        This function is to build augmented matrix from 'matrix' and 
        'constants', which represents linear system of equations like
                  a1x + b1y = c1 ->  [a1 b1], [c1]
                  a2x + b2y = c2 ->  [a2 b2], [c2]
        Then transform it to
                  x = c1/a1 - b1/a1y -> [0 -b1/a1], [c1/a1]
                  y = c1/b2 - a2/b2x -> [-a2/b2 0], [c1/b2]
        that each row of the transformed matrix represents the composition
        of corresponding variable.

        paras:
          matrix: np.ndarray, n x n matrix.
          constants: np.ndarray, nx1 array consists of constants of system.

        return :
          matrix: np.ndarray, n x n matrix.
          constants: np.ndarray, nx1 array
    """

    matrix = np.array(matrix).astype(float)
    constants = constants.astype(float)
    for i, row in enumerate(matrix):
        constants[i] = constants[i] / row[i]
        row /= row[i]
        row[i] = 0
    ones = np.eye(matrix.shape[0], matrix.shape[0]) * -1.0
    matrix = matrix @ ones
    return matrix, constants


def jacobi_iteration_core(matrix, constants, tolerance= 0.00001, 
                          tolerance_function=None):
    """
        This function uses Jacobi iteration to find solution. It assumes the
        matrix has well-permutated rows and is full-rank that guarantees to
        find the solution.

        paras:
          matrix: np.ndarray, n x n matrix
          constans: np.ndarray, nx1 dimension, the constants of the system.
          tolerance: float, the tolerance between consecutive iterations.
          tolerance_function: function returns bool, the function to compute
            tolerance which must take two n x 1 np.ndarray of candidate 
            solutions, a new one and a previous one, and 'tolerance' into
            calculation. Returns True if 'tolerance' is satisfied, otherwise
            returns False.
    """

    # prepare system for iterations
    matrix, constants = _matrix_preprocess_for_iteration(matrix, constants)

    # determine tolerance function
    if tolerance_function is None:
        # this lambda will crash when the true value is 0
        tolerance_function = lambda x, y, t: (np.absolute((x-y)/y) <= t).all()
        
    # iterations
    new_sol = np.zeros((matrix.shape[0], 1))
    old_sol = np.zeros((matrix.shape[0], 1)) + tolerance
    count = 0
    while not tolerance_function(new_sol, old_sol, tolerance):
        old_sol = new_sol
        new_sol = matrix @ new_sol + constants
        count += 1

    print(f'----Iteration count: {count}----')
    return new_sol


def gauss_seidel_iteration_core(matrix, constants, tolerance= 0.00001, 
                                tolerance_function=None):
    """
        This function uses Gauss-Seidel iteration to find the solution. It 
        assumes the matrix has well-permutated rows and is full-rank that
        guarantees to find the solution.

        paras:
          matrix: np.ndarray, n x n matrix
          constans: np.ndarray, nx1 dimension, the constants of the system.
          tolerance: float, the tolerance between consecutive iterations.
          tolerance_function: function returns bool, the function to compute
            tolerance which must take two n x 1 np.ndarray of candidate 
            solutions, a new one and a previous one, and 'tolerance' into
            calculation. Returns True if 'tolerance' is satisfied, otherwise
            returns False.
    """

    # prepare system for iterations
    matrix, constants = _matrix_preprocess_for_iteration(matrix, constants)

    # determine tolerance function
    if tolerance_function is None:
        # this lambda will crash when the true value is 0
        tolerance_function = lambda x, y, t: (np.absolute((x-y)/y) <= t).all()
        
    # iterations
    new_sol = np.zeros((matrix.shape[0], 1))
    old_sol = np.zeros((matrix.shape[0], 1)) + tolerance
#    count = 0
    while not tolerance_function(new_sol, old_sol, tolerance):
        old_sol = new_sol
        new_sol = np.array(new_sol)
        for i, row in enumerate(matrix):
            new_sol[i, 0] = row @ new_sol + constants[i]
#        count += 1

#    print(f'----Iteration count: {count}----')
    return new_sol


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
    s0 = (np.array([[5, 1, -1], [1, -5, 2], [1, -2, 10]]),
          np.array([14, -9, -30])[:, np.newaxis])

    system = s0
    print(jacobi_iteration_core(system[0], system[1]))
    print(gauss_seidel_iteration_core(system[0], system[1]))
#    matrix = a3
#    print(f'original matrix:\n{matrix}')
#    s = solve_linear_equations(matrix)
#    print('---solutions by "solve_linear_equations"---')
#    print(f'basic:\n{s[0]}')
#    print(f'free:\n{s[1]}')
#    basic, free = lu_linear_solution(matrix[:, :-1], matrix[:, -1])
#    print('---solutions by LU---')
#    print(f'basic:\n{basic}')
#    print(f'free:\n{free}')
