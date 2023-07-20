import numpy as np
from reduced_row_echelon_form import reduced_row_echelon_form
from reduced_row_echelon_form import leading_nonzero


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
    matrix = reduced_row_echelon_form(matrix)

    # solve each variable in terms of variables and constant from top of
    # the matrix to the bottom. If contradiction occur, the system is 
    # inconsistent, then returns None.
    for i, row in enumerate(matrix):
        leading_index = leading_nonzero(matrix, i)
        # contradiction occurs: 0 = 1
        if leading_index == total_elements:
            return [], []
        # 0 = 0 occurs, no need to process further
        if leading_index == -1:
            break

        # update variable with terms other than itself
        variables[leading_index] -= row
        variables[leading_index][-1] *= -1.0

    # solve each variable in back-substitution style and update each 
    # corresponding element in the equation of the variable.
    for variable, index in zip(variables[::-1], range(len(variables)-1, -1, -1)):
        # search for non-zero coefficient in variable's equation
        for i in range(total_elements):
            # add variable with basic variable times the corresponding
            # coefficient to the constant part of the variable's equation.
            if variable[i] and not isinstance(variables[i], type(variable)):
                variable[-1] += variable[i] * variables[i]
                variable[i] = 0

    free_vectors = []
    for i in range(total_elements):
        if any(variables[:, i]):
            free_vectors.append(variables[:, i])
    basic_vector = variables[:, -1]

    return basic_vector, free_vectors


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
    matrix = a9
    print(f'original matrix:\n{matrix}')
    s = solve_linear_equations(matrix)
    print(f'solutions:\n{s[0]}\n{s[1]}')
