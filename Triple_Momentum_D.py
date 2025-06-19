# Triple Momentum Method 
# Created by He HAO Ph.D. student (ist194693) with Github link (https://github.com/HAOHE123)

import numpy as np
import sympy as sy

# Easy gradient
# def gradient(scalar_function, variables):
#     matrix_scalar_function = sy.Matrix([scalar_function])
#     return matrix_scalar_function.jacobian(variables)

# Function for Gradient_Descent
def Triple(L, M, x, X, DIMC, x_position, x_previous, obj, iteras):
    # "Optimal" Parameters

    # rho = 1-sqrt(1/k)   k=L/M
    rho = 1 - np.sqrt(M/L)
    # print(rho)
    # Parameters
    alpha = rho**2 / (2-rho)
    beta = (1+rho) / L
    gamma = rho**2 / ( (1+rho)*(2-rho) )
    sigma = rho**2/ (1-rho**2)

    #  # Values for Mexico case without Adaptive
    # beta = 10000
    # alpha = 0.15
    # gamma = 0.1
    # sigma = 0.05
    
    # Find the gradient of the obj
    obj_matrix = sy.Matrix([obj])
    gradient_f = obj_matrix.jacobian(X).T

    # positions
    x_gra = np.zeros((DIMC,iteras))
    x_gra = sy.Matrix(x_gra)
    x_gra[:,0] = x_position


    z_gra = np.zeros((DIMC,iteras))
    z_gra = sy.Matrix(z_gra)
    # The first step is special (no k-1)
    
    # The variable y = (1+apha)*x
    y = (1+gamma) * x_gra[:, 0] - gamma * x_previous
    gradient_evaluated = gradient_f.subs({x: y}).evalf()
    x_position = (1+alpha) * x_gra[:,0] - alpha * x_previous + beta * gradient_evaluated

    # no velocity initially
    # y = (1+gamma) * x_gra[:, 0] 
    # gradient_evaluated = gradient_f.subs({x: y}).evalf()
    # x_position = (1+alpha) * x_gra[:,0] + beta * gradient_evaluated

    x_gra[:, 1] = x_position

    # x_position = (1+alpha)*x_gra[:,0] + beta*gradient_f

    # x_position = x_position.subs({x: y}).evalf()

    
    # print(x_position)

    z_gra[:,0] = x_gra[:,0]
    z_gra[:,1] =(1+sigma)*x_gra[:,1] - sigma*x_gra[:,0]
    

    for i in range(1,iteras-1):
        # Take one step ahead
        # x = x_gra[:,i]
        # Update y
        y = (1 + gamma) * x_gra[:, i] - gamma * x_gra[:, i-1]

        # Evaluate the gradient at y
        gradient_evaluated = gradient_f.subs({x: y}).evalf()

        x_position = (1+alpha)*x_gra[:, i] - alpha*x_gra[:,i-1] + beta*gradient_evaluated

        # # The variable y = (1+apha)*x_k-x_k-1
        # y = (1+gamma)*x_gra[:,i] - gamma*x_gra[:,i-1] 

        # x_position = x_position.subs({ x: y }).evalf()
        # print(x_position)
        # Update x_gra
        x_gra[:, i+1] = x_position

        # Update z_gra
        z_gra[:, i+1] = (1 + sigma) * x_gra[:, i+1] - sigma * x_gra[:, i]


    return [x_gra,z_gra]
