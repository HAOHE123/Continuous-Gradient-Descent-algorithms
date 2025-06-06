# Nestrov Method in Discrete
# Created by He HAO Ph.D. student (ist194693) with Github link (https://github.com/HAOHE123)

import numpy as np
import sympy as sy

# Easy gradient
# def gradient(scalar_function, variables):
#     matrix_scalar_function = sy.Matrix([scalar_function])
#     return matrix_scalar_function.jacobian(variables)

# Function for Gradient_Descent
def Nestrov(L, M, x, X, DIMC, x_position, obj, iteras):
    # "Optimal" Parameters
    beta = 1/L 
    alpha = (np.sqrt(L)-np.sqrt(M)) / (np.sqrt(L)+np.sqrt(M))
    
    # Find the gradient of the obj
    gradient_f = obj.jacobian(X).T

    # positions
    x_gra = np.zeros((DIMC,iteras))
    x_gra = sy.Matrix(x_gra)
    x_gra[:,0] = x_position

    # The first step is special (no k-1)
    
    x_position = (1+alpha)*x_gra[:,0] - beta*gradient_f

    # The variable y = (1+apha)*x
    y = (1+alpha)*x_gra[:,0]
    # print(y)

    x_position = x_position.subs({x: y})
    # print(x_position)
    x_gra[:,1] = x_position[:,0]


    for i in range(1,iteras-1):
        # Take one step ahead
        # x = x_gra[:,i]

        x_position = (1+alpha)*x_position - alpha*x_gra[:,i-1] - beta*gradient_f

        # The variable y = (1+apha)*x_k-x_k-1
        y = (1+alpha)*x_gra[:,i] - alpha*x_gra[:,i-1] 

        x_position = x_position.subs({ x: y })
        # print(x_position)
        x_gra[:,i+1] = x_position[:,0]


    return x_gra
