# Nestrov Method in Continuous
# Created by He HAO Ph.D. student (ist194693) with Github link (https://github.com/HAOHE123)
import json
import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import eig

# Save the result as json
def save_data_to_json(file_name, data):
    with open(file_name, 'w') as file:
        json.dump(data, file)

# Function for Gradient_Descent
def Nestrov_C(pi, L, M, Q, e, x0, DIMC, t_span, t_step):
    # "Optimal" Parameters
    k = L/M
    beta = pi/ M
    # beta = (pi^2) / M
    alpha = np.sqrt(4*beta*L)
    
    beta = pi
    alpha = np.sqrt(4*beta*M)


    # beta = beta*1000
    Q = np.matrix(Q)

    # Initial position
    X0 = np.block([
        [x0],
        [np.zeros((DIMC, 1))]
    ])
    X0 = np.ndarray.flatten(X0)  # change X0 to 1-dimension

    # Parameter e
    e = np.block([
        [e],
        [np.zeros((DIMC, 1))]
    ])
    
    # The matrix parameters
    A = np.block([
        [np.zeros((DIMC,DIMC)),  np.eye(DIMC)],
        [np.zeros((DIMC,DIMC)),  -alpha*np.eye(DIMC) ]
    ])
    
    B = np.block([
        [np.zeros((DIMC,DIMC))],
        [-beta*np.eye(DIMC)]
    ])

    C = np.block([
        [ np.eye(DIMC),   np.zeros((DIMC,DIMC)) ] 
    ])

    MAtrix = np.array(A + B*Q*C)
    [tt,aa] = eig(MAtrix)
    print(tt)
    dX = lambda t, X: np.dot(MAtrix, X) + np.ndarray.flatten(e)
    t_eval = np.arange(0, t_span+t_step, t_step)
    
    # C -- ODE function
    sol = solve_ivp(dX, [0, t_span], X0, t_eval=t_eval)

    # The format is X = [[x],[x_dot]]
    x_gra = sol.y
    x_gra = x_gra[0:2,:]

    # Organize the data in a dictionary
    # data_to_save = {
    #     "alpha": alpha,
    #     "beta": beta,
    #     "Q": Q.tolist(),
    #     "H": (np.transpose(Q)*Q).tolist()
    # }
    data_to_save = {
        "alpha": alpha,
        "beta": beta,
        "Q": Q.tolist()
    }

    # Save the data
    file_name = 'Nestrov_C_result.json'
    save_data_to_json(file_name, data_to_save)

    return x_gra