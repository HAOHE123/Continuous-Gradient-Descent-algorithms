# Nestrov Method in Discrete
# Created by He HAO Ph.D. student (ist194693) with Github link (https://github.com/HAOHE123)

import numpy as np
import sympy as sy

# Easy gradient
# def gradient(scalar_function, variables):
#     matrix_scalar_function = sy.Matrix([scalar_function])
#     return matrix_scalar_function.jacobian(variables)

# Function for Gradient_Descent
def Nestrov(L, M, x, X, DIMC, x_position, x_previous, obj, iteras):
    # need to count one more for the 1st
    iteras = iteras + 1
    # "Optimal" Parameters
    beta = 1/L 
    alpha = (np.sqrt(L)-np.sqrt(M)) / (np.sqrt(L)+np.sqrt(M))

    # # Values for Mexico case without Adaptive
    # beta = 10000
    # alpha = 0.25
    
    # Find the gradient of the obj
    obj_matrix = sy.Matrix([obj])
    gradient_f = obj_matrix.jacobian(X).T

    # positions
    x_gra = np.zeros((DIMC,iteras))
    x_gra = sy.Matrix(x_gra)
    x_gra[:,0] = x_position

    # The first step is special (no k-1)
    y = (1+alpha) * x_gra[:, 0] - alpha * x_previous  # Initially, y is just x_position
    
    # # no velocity initially
    # y = (1+alpha) * x_gra[:, 0]  # Initially, y is just x_position

    gradient_evaluated = gradient_f.subs({x: y}).evalf()
    x_position = y + beta * gradient_evaluated
    x_gra[:, 1] = x_position

    # The variable y = (1+apha)*x
    # y = (1+alpha)*x_gra[:,0]

    # x_position = x_position.subs({x: y}).evalf()
    # print(x_position)
    # x_gra[:,1] = x_position


    for i in range(1, iteras-1):
        # Take one step ahead
        # x = x_gra[:,i]
        # The variable y = (1+apha)*x_k-x_k-1
        y = (1+alpha)*x_gra[:,i] - alpha*x_gra[:,i-1] 

        gradient_evaluated = gradient_f.subs({x: y}).evalf()

        x_position = y + beta*gradient_evaluated

        

        # x_position = x_position.subs({ x: y }).evalf()
        # print(x_position)
        x_gra[:,i+1] = x_position


    return x_gra[:,1:]
