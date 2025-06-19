# Triple Momentum Method in Continuous
# Created by He HAO Ph.D. student (ist194693) with GitHub link (https://github.com/HAOHE123)

import numpy as np
import sympy as sy
from scipy.integrate import solve_ivp


def Three_Variables_C(pi, L, M, x, X, D, x0, obj, t_span, t_val):
    # "Optimal" Parameters
    alpha = (2 * np.sqrt(pi * L * M) * (np.sqrt(L) - np.sqrt(M))) / (L - M)
    beta = pi
    gamma = (2 / np.sqrt(pi)) * (np.sqrt(L) - np.sqrt(M)) / (L - M)

    if M < 1e-5:
        sigma = 1.1
    else:
        sigma = (2) / ((-alpha - beta * gamma * M) / 2)**2

    # ensure x0 is a numpy array
    x0 = np.array(x0).flatten()
    DIMC = D

    # define the gradient function
    # we want to maximize => negative of obj
    obj_matrix = -sy.Matrix([obj])
    gradient_f = obj_matrix.jacobian(X).T
    gradient_f = sy.lambdify(X, gradient_f, 'numpy')

    def triple_momentum_ode(t, Y):
        x_ = Y[:DIMC]    # position
        v_ = Y[DIMC:2*DIMC]  # velocity
        y_ = x_ + gamma*v_

        grad_ = np.array(gradient_f(*y_)).flatten()
        dxdt = v_
        dvdt = -alpha*v_ - beta*grad_
        return np.concatenate([dxdt, dvdt])

    # initial conds
    v0 = np.zeros_like(x0)
    Y0 = np.concatenate([x0, v0])

    # time mesh
    t_eval = np.linspace(0, t_span, num=1000)

    # solve ODE
    sol = solve_ivp(triple_momentum_ode, [0,t_span], Y0, t_eval=t_eval)

    # original code:
    x_gra = sol.y[:DIMC, :]       # positions shape (D, #samples)
    v_gra = sol.y[DIMC:2*DIMC, :] # velocities shape (D, #samples)

    # compute 'o' over time
    o_gra = x_gra + sigma*v_gra

    # >>>> Optional: compute a_gra from ODE definition, no numeric differ:
    a_gra = np.zeros_like(v_gra)  # same shape
    for i in range(v_gra.shape[1]):
        # x(t_k) = x_gra[:, i]
        # v(t_k) = v_gra[:, i]
        x_ = x_gra[:, i]
        v_ = v_gra[:, i]
        y_ = x_ + gamma*v_
        grad_ = np.array(gradient_f(*y_)).flatten()
        a_ = -alpha*v_ - beta*grad_   # from dvdt
        a_gra[:, i] = a_
    # <<<< End

    # returning them all, minimal new line at the end:
    return x_gra, v_gra, a_gra, o_gra
