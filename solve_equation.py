import numpy as np
from reduced_row_echelon_form import reduced_row_echelon_form
from reduced_row_echelon_form import leading_nonzero


class _Vector:
    pass


class _Variable:
    """
        The core is 'self.equation', which represents the linear
        combinations of the variable by other variables. The last one of the
        equation is the constant. Each element in the equation refers to the
        coefficient of the corresponding variable. For example:
        
                                v1 v2 v3  v4  const
        variable v's equation = [0, 0, 2, -3, 10], then
        v = 0*v1 + 0*v2 + 2*v3 - 3*v4 + 10

                          v1  v2 v3  const
        If v's equation = [0 , 1, 0, np.nan], it means that v is actually
        v2 and it's a free variable, since the constant is np.nan (there's
        no proper object representing free variable, so use np.nan as an
        alternative).
    """

    parameter = 'v'
    number = 0

    def __init__(self, total_elements):
        self.number = self.__class__.number
        self.__class__.number += 1
        self.equation = np.zeros(total_elements)
        self.equation[-1] = np.nan
        self.equation[self.number] = 1

    def __repr__(self):
        return f'{self.__class__.parameter}{self.number}={self.equation}'

    def __getitem__(self, key):
        return self.equation[key]

    def __setitem__(self, key, value):
        self.equation[key] = value

    def __isub__(self, other):
        self.equation -= other
        return self



def solve_linear_equations(matrix):
    total_elements = matrix.shape[1] - 1
    variables = [_Variable(total_elements+1) for _ in range(total_elements)]
    matrix = reduced_row_echelon_form(matrix)
    print(f'reduced matrix:\n{matrix}')
    print(f'before process:\n{variables}\n')

    # solve each variable in terms of variables and constant from top of
    # the matrix to the bottom. If contradiction occur, the system is 
    # inconsistent and then returns None.
    for i, row in enumerate(matrix):
        leading_index = leading_nonzero(matrix, i)
        # contradiction occurs: 0 = 1
        if leading_index == total_elements:
            return None
        # 0 = 0 occurs
        if leading_index == -1:
            break

        # the variable chosen by leading_index is processed
        # the variable unselected is free variable
        variables[leading_index][-1] = 0
        variables[leading_index] -= row
        variables[leading_index][-1] *= -1.0

    # solve each variable in back-substitution style. If any variable can
    # be an exact number, replace corresponding position in variables with
    # that number. Otherwise, update each corresponding element in equation
    # of the variable.
    for variable, index in zip(variables[::-1], range(len(variables)-1, -1, -1)):
        if variable[-1] != np.nan:
            # search for non-zero coefficient in variable's equation
            for i in range(total_elements):
                # add variable with known number times corresponding
                # coefficient to the constant at the end of variable's 
                # equation.
                if variable[i] and not isinstance(variables[i], type(variable)):
                    variable[-1] += variable[i] * variables[i]
                    variable[i] = 0
            if not any(variable[:total_elements]):
                variables[index] = variable[-1]

    return variables
    


if __name__ == '__main__':
    a1 = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0]])
    a2 = np.array([[0, 0, 3], [2, 2, 3]])
    a3 = np.array([[1, -2, -1, 3], [3, -6, -5, 3], [2, -1, 1, 0]])
    a4 = np.array([[0, 0, 7], [0, 0, 0], [3, 0, 3]])
    a5 = np.array([[0, 0, 11, 0], [0, 0, 0, 0], [2, 0, 8, 6], [0, 7, 7, 14]])
    a6 = np.array([[1, -3, 0, 2, 0, 7], [0, 0, 1, 6, 0, 9], [0, 0, 0, 0, 1, 2]])
    a8 = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    a9 = np.array([[1, 2, -1, 2, 1, 2], [-1, -2, 1, 2, 3, 6], [2, 4, -3, 2, 0, 3], [-3, -6, 2, 0, 3, 9]])
    matrix = a6
    print(matrix)
    s = solve_linear_equations(matrix)
    print(s)
