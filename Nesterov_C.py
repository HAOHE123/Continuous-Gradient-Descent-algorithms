# Nesterov Method in Continuous-Time
# Created by He HAO Ph.D. student (ist194693)
import numpy as np
import sympy as sy
from scipy.integrate import solve_ivp

def Nesterov_C(pi, L, M, x, X, D, x0, obj, t_span, t_val):
    # "Optimal" Parameters
    beta = pi  # Given parameter
    alpha = np.sqrt(4 * beta * M)  # Computed based on beta and M

    # Ensure initial position is a flat array
    x0 = np.array(x0).flatten()
    DIMC = D  # Dimension of the problem

    # Define the gradient function using sympy
    # Note: we want to maximize the objective function, so we need to minimize the negative of the objective function.
    obj_matrix = -sy.Matrix([obj])
    gradient_f = obj_matrix.jacobian(X).T  # Compute the symbolic gradient
    # Convert the symbolic gradient to a numerical function
    gradient_f = sy.lambdify(X, gradient_f, 'numpy')

    # Define the ODE system
    def heavy_ball_ode(t, Y):
        x = Y[:DIMC]       # Position vector x
        v = Y[DIMC:]       # Velocity vector v = dx/dt
        grad = np.array(gradient_f(*x)).flatten()  # Compute gradient at x

        # Compute derivatives
        dxdt = v
        dvdt = -alpha * v - beta * grad

        return np.concatenate([dxdt, dvdt])

    # Initial conditions: position x0 and initial velocity v0 = 0
    v0 = np.zeros_like(x0)
    Y0 = np.concatenate([x0, v0])

    # Time evaluation points
    t_eval = np.linspace(0, t_span, num= 1000)

    # Solve the ODE system
    sol = solve_ivp(heavy_ball_ode, [0, t_span], Y0, t_eval=t_eval)

    # Extract position and velocity over time
    x_gra = sol.y[:DIMC, :]  # Positions
    v_gra = sol.y[DIMC:, :]  # Velocities

    return x_gra
