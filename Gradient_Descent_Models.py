# Gradient Descent Models
# Created by He HAO Ph.D. student (ist194693) with Github link (https://github.com/HAOHE123)
import json
import scipy.io
from numpy.linalg import eig, qr
import numpy as np
import sympy as sy
import numpy.linalg as la
from numpy.linalg import eig 
import matplotlib.pyplot as plt

# import gradient methods funs
import Nestrov_D as ND
import Triple_Momentum_D as TD
import Nestrov_C as NC
import Three_Variables_C as ThrVC

import psutil, os, time
def Models(Dis_M,Con_M):
# Predefined values
    # If draw the plot, DIMC == 2
    DIMC = 2
    # Generate a positive definite symmetric matrix A
    A = np.random.uniform(low=0.1, high=1.0, size=(DIMC,DIMC))
    A = 1/2 * (A+A.T)
    # A[1:3,:] = 4*A[1:3,:]
    # Make sure it is positive
    # A += DIMC*np.eye(DIMC);
    # Probably fail but more random A
    A += np.array(0.2)*np.eye(DIMC);
    print(A)
    # A = np.matrix([[0.79810017, 0.612118  ],
    # [0.612118,   0.63865655]])

    # A = np.matrix([[2, 1 ],
    # [1,   2]])
    # A = np.array([
    # [2.69764539, 0.25562505, 0.3824604,  0.67047799, 0.17172448, 0.90468271],
    # [1.02250018, 5.60948485, 1.6527293,  1.77989591, 2.11039157, 1.43500383],
    # [1.5298416,  1.6527293,  2.78450088, 2.08019266, 1.17733925, 3.75274048],
    # [0.67047799, 0.44497398, 0.52004816, 3.41263637, 0.46765566, 0.53881662],
    # [0.17172448, 0.52759789, 0.29433481, 0.46765566, 2.80097102, 0.33168664],
    # [0.90468271, 0.35875096, 0.93818512, 0.53881662, 0.33168664, 2.96368505]
    # ])
    # Remaining values


    # Better way to generate the matrix A
    # Step 1: Choose the desired eigenvalues (ensure they are > 0 for positive definiteness)
    # Here, for instance, pick them randomly between (0.1, 2.0):
    # desired_eigs = np.random.uniform(low=0.1, high=1, size=DIMC)
    # # desired_eigs[0] = 0.3
    # # Step 2: Generate a random matrix and perform QR factorization to get an
    # # orthonormal set of vectors (the columns of Q).
    # random_matrix = np.random.randn(DIMC, DIMC)
    # D, _ = qr(random_matrix)  # Q is orthonormal

    # # Step 3: Construct A = Q * diag(eigenvalues) * Q^T
    # Lambda = np.diag(desired_eigs)
    # A = D @ Lambda @ D.T  # By construction, A is symmetric positive definite

    # Just to verify:
    # (e_vals, e_vect) = eig(A)
    # print("Eigenvalues:", e_vals)  # Should all be positive

    # diagonal A
    # Step 1: Generate random eigenvalues, all > 0
    desired_eigs = np.random.uniform(low=0.1, high=1.0, size=DIMC)

    # Step 2: Construct a purely diagonal matrix
    A = np.diag(desired_eigs)

    (e_vals, e_vect) = eig(A)
    L = max(e_vals)
    M = min(e_vals)

    # Global values
    global e
    e = np.zeros((DIMC, 1))

    global c
    c = np.zeros((1, 1))
    
    global Q
    Q = A

    # Variable x and its matrix format X
    x = sy.MatrixSymbol('x', DIMC, 1)
    X = sy.Matrix(x)
    x0 = 2*np.ones((DIMC, 1))
    
    # Objective function
    obj = 1/2*X.T*A*X - e.T*X + c

    # Find the optimal result
    gradient_f = obj.jacobian(X).T
    global x_opt
    x_opt = sy.solve(gradient_f,X)
    x_opt = x_opt.items()
    x_opt = list(x_opt)
    x_opt = np.array(x_opt)
    x_opt = x_opt[:,1]
    x_opt = np.reshape(x_opt,(DIMC,1))
    print('Opt_Value:',x_opt)

    # Discrete Model
    if Dis_M:
        # iterations
        global iterations
        iterations = 20

        # Gradient Descent
        global x_ND 
        x_ND = ND.Nestrov(L,M,x,X,DIMC,x0,obj,iterations)

        global x_tD 
        global x_TD 
        [x_tD, x_TD] = TD.Triple(L,M,x,X,DIMC,x0,obj,iterations)


    # Continuous Models
    if Con_M:
        # Parameter Pi
        pi = 3000

        # time span 
        global t_span
        t_span = 2

        # Three variable method and Triple Momentum method in continuous-time
        global x_ThrVC, x_FourVC 
        [x_ThrVC, x_FourVC ]= ThrVC.Three_Variables_C(pi, L, M, Q, e, x0, DIMC, t_span, t_step)
    


# Plot the results for the discrete case
def Gradient_Plot_D():

    # Import all the data
    global Q
    global e
    global c
    global x_GD 
    global x_HD 
    global x_ND 
    global x_tD
    global x_TD
    global iterations
    n_points = 101
    u = np.linspace(-1, 2, n_points)
    x, y = np.meshgrid(u, u)

    X = np.vstack([x.flatten(), y.flatten()])
    f_x = np.dot(np.dot(X.T, Q), X) - np.dot(e.T, X) + c
    f_x = np.diag(f_x).reshape(n_points, n_points)


    # Plot figures
    plt.figure(0)
    # Create six sub-figures

    plt.subplot(2, 3, 1)
    # Contour
    cs = plt.contour(x, y, f_x, levels=15)
    # Line
    for i in range(0,iterations-1):
        plt.plot( [ x_GD[0,i], x_GD[0,i+1] ] , [ x_GD[1,i], x_GD[1,i+1] ] ,'bo-')
    # Labels
    plt.clabel(cs,inline=1,fontsize=9)
    plt.grid(True)
    plt.axis('scaled')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Gradient Descent in Discrete Time')
   


    plt.subplot(2, 3, 2)
    # Contour
    cs = plt.contour(x, y, f_x, levels=15)
    # Line
    for i in range(0,iterations-1):
        plt.plot( [ x_HD[0,i], x_HD[0,i+1] ] , [ x_HD[1,i], x_HD[1,i+1] ] ,'bo-')
    plt.clabel(cs,inline=1,fontsize=9)
    plt.grid(True)
    plt.axis('scaled')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Heavy-Ball in Discrete Time')



    plt.subplot(2, 3, 3)
    # Contour
    cs = plt.contour(x, y, f_x, levels=15)
    # Line
    for i in range(0,iterations-1):
        plt.plot( [ x_ND[0,i], x_ND[0,i+1] ] , [ x_ND[1,i], x_ND[1,i+1] ] ,'bo-')
    # Labels
    plt.clabel(cs,inline=1,fontsize=9)
    plt.grid(True)
    plt.axis('scaled')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Nestrov Method in Discrete Time')



    plt.subplot(2, 3, 4)
    # Contour
    cs = plt.contour(x, y, f_x, levels=15)
    # Line
    for i in range(0,iterations-1):
        plt.plot( [ x_tD[0,i], x_tD[0,i+1] ] , [ x_tD[1,i], x_tD[1,i+1] ] ,'bo-')
    # Labels
    plt.clabel(cs,inline=1,fontsize=9)
    plt.grid(True)
    plt.axis('scaled')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Triple Variables in Discrete Time')
 

    plt.subplot(2, 3, 5)
    # Contour
    cs = plt.contour(x, y, f_x, levels=15)
    # Line
    for i in range(0,iterations-1):
        plt.plot( [ x_TD[0,i], x_TD[0,i+1] ] , [ x_TD[1,i], x_TD[1,i+1] ] ,'bo-')
    # Labels
    plt.clabel(cs,inline=1,fontsize=9)
    plt.grid(True)
    plt.axis('scaled')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Triple Momentum in Discrete Time')
    plt.show()

    


def Gradient_Plot_C():
    # Import all the data
    global Q
    global e
    global c
    global x_NC
    global x_TVC
    global x_ThrVC, x_FourVC 
    global t_span
    global t_step
   
    n_points = int(1/t_step)*t_span + 1
    u = np.linspace(-1, 2, n_points)
    x, y = np.meshgrid(u, u)

    X = np.vstack([x.flatten(), y.flatten()])
    f_x = np.dot(np.dot(X.T, Q), X) - np.dot(e.T, X) + c
    f_x = np.diag(f_x).reshape(n_points, n_points)
    
    # Plot figures
    plt.figure(1)
    # Create six sub-figures

    plt.subplot(2, 2, 1)
    # Contour
    cs = plt.contour(x, y, f_x, levels=15)
    # Line
    for i in range(0,n_points-1):
        plt.plot( [ x_NC[0,i], x_NC[0,i+1] ] , [ x_NC[1,i], x_NC[1,i+1] ] ,'bo-')
    # Labels
    plt.clabel(cs,inline=1,fontsize=9)
    plt.grid(True)
    plt.axis('scaled')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    # plt.title('Continuous-Time Nestrov Method')
    plt.title('Continuous Heavy-Ball')

    plt.subplot(2, 2, 2)
    # Contour
    cs = plt.contour(x, y, f_x, levels=15)
    # Line
    for i in range(0,n_points-1):
        plt.plot( [ x_TVC[0,i], x_TVC[0,i+1] ] , [ x_TVC[1,i], x_TVC[1,i+1] ] ,'bo-')
    # Labels
    plt.clabel(cs,inline=1,fontsize=9)
    plt.grid(True)
    plt.axis('scaled')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    # plt.title('Two Parameters Multi-Buffer Approach')
    plt.title('Continuous Nesterov')

    plt.subplot(2, 2, 3)
    # Contour
    cs = plt.contour(x, y, f_x, levels=15)
    # Line
    for i in range(0,n_points-1):
        plt.plot( [ x_ThrVC[0,i], x_ThrVC[0,i+1] ] , [ x_ThrVC[1,i], x_ThrVC[1,i+1] ] ,'bo-')
    # Labels
    plt.clabel(cs,inline=1,fontsize=9)
    plt.grid(True)
    plt.axis('scaled')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    # plt.title('Innovating Multi-Buffer Approach')
    plt.title('Continuous Triple-Momentum(without o)')

    plt.subplot(2, 2, 4)
    # Contour
    cs = plt.contour(x, y, f_x, levels=15)
    # Line
    for i in range(0,n_points-1):
        plt.plot( [ x_FourVC[0,i], x_FourVC[0,i+1] ] , [ x_FourVC[1,i], x_FourVC[1,i+1] ] ,'bo-')
    # Labels
    plt.clabel(cs,inline=1,fontsize=9)
    plt.grid(True)
    plt.axis('scaled')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    # plt.title('Triple Momentum in Multi-Buffer Approach')
    plt.title('Continuous Triple Momentum')
   

    plt.show()
    



def Error_Plot_D():
    # Import all the data
    global x_opt
    global x_GD 
    global x_HD 
    global x_ND 
    global x_tD
    global x_TD
    global iterations

    # Plot figures
    plt.figure(2)
    # Error
    x_lin = np.linspace(0,iterations,iterations)
    x_GD_e = np.zeros(iterations)
    x_HD_e = np.zeros(iterations)
    x_ND_e = np.zeros(iterations)
    x_tD_e = np.zeros(iterations)
    x_TD_e = np.zeros(iterations)

    for i in range(0,iterations):

        e_value = x_GD[:,i] - x_opt
        e_value = np.array(e_value, dtype=np.float64)
        x_GD_e[i] = la.norm(e_value)

        e_value = x_HD[:,i] - x_opt
        e_value = np.array(e_value, dtype=np.float64)
        x_HD_e[i] = la.norm(e_value)

        e_value = x_ND[:,i] - x_opt
        e_value = np.array(e_value, dtype=np.float64)
        x_ND_e[i] = la.norm(e_value)

        e_value = x_tD[:,i] - x_opt
        e_value = np.array(e_value, dtype=np.float64)
        x_tD_e[i] = la.norm(e_value)

        e_value = x_TD[:,i] - x_opt
        e_value = np.array(e_value, dtype=np.float64)
        x_TD_e[i] = la.norm(e_value)

    # plt.plot(x_lin, x_GD_e, c="red", lw=3, linestyle="dashdot", label="Gradient Descent")
    plt.plot(x_lin, x_GD_e, label="Gradient Descent")
    plt.plot(x_lin, x_HD_e, label="Heavy-Ball")
    plt.plot(x_lin, x_ND_e, label="Nestrov Method")
    plt.plot(x_lin, x_tD_e, label="Triple Variables")
    plt.plot(x_lin, x_TD_e, label="Triple Momentum")

    # Setting the x-axis limit
    plt.xlim(0, iterations)

    # Labels
    plt.grid(True)
    plt.xlabel('k-steps')
    plt.ylabel('Error Estimation $\epsilon$')
    plt.yscale('log')
    plt.title('Discrete-Time Error Estimation')
    plt.legend(loc='upper right')
    plt.show()




def Error_Plot_C():
    # Import all the data
    global x_opt
    global x_NC
    global x_TVC
    global x_ThrVC, x_FourVC
    global t_span
    global t_step
   
    n_points = int((t_span/t_step) + 1)

    # Plot figures
    plt.figure(3)
    # Error
    x_lin = np.linspace(0,t_span,n_points)
    x_NC_e = np.zeros(n_points)
    x_TVC_e = np.zeros(n_points)
    x_ThrVC_e = np.zeros(n_points)
    x_FourVC_e = np.zeros(n_points)

    for i in range(0,n_points):

        e_value = x_NC[:,i] - x_opt
        e_value = np.array(e_value, dtype=np.float64)
        x_NC_e[i] = la.norm(e_value)

        e_value = x_TVC[:,i] - x_opt
        e_value = np.array(e_value, dtype=np.float64)
        x_TVC_e[i] = la.norm(e_value)

        e_value = x_ThrVC[:,i] - x_opt
        e_value = np.array(e_value, dtype=np.float64)
        x_ThrVC_e[i] = la.norm(e_value)

        e_value = x_FourVC[:,i] - x_opt
        e_value = np.array(e_value, dtype=np.float64)
        x_FourVC_e[i] = la.norm(e_value)

    # plt.plot(x_lin, x_GD_e, c="red", lw=3, linestyle="dashdot", label="Gradient Descent")
    # plt.plot(x_lin, x_NC_e, label="Continuous-Time Nestrov Method")
    # plt.plot(x_lin, x_TVC_e, label="Two Parameters Multi-Buffer Approach")
    # plt.plot(x_lin, x_ThrVC_e, label="Innovating Multi-Buffer Approach")
    # plt.plot(x_lin, x_FourVC_e, label="Triple Momentum in Multi-Buffer Approach")
    plt.plot(x_lin, x_NC_e, label="Continuous Heavy-Ball")
    plt.plot(x_lin, x_TVC_e, label="Continuous Nestrov")
    plt.plot(x_lin, x_ThrVC_e, label="Continuous Triple-Momentum(without o)")
    plt.plot(x_lin, x_FourVC_e, label="Continuous Triple Momentum")

    data_to_save = {
        "x_lin": x_lin,
        "x_NC_e": x_NC_e,
        "x_TVC_e": x_TVC_e,
        "x_ThrVC_e": x_ThrVC_e,
        "x_FourVC_e": x_FourVC_e
    }

# Save the data
    file_name = 'position_result.mat'
    scipy.io.savemat(file_name, data_to_save)

    plt.xlim(0, t_span)
    # Labels
    plt.grid(True)
    plt.xlabel('Time(s)')
    plt.ylabel('Error Estimation $\epsilon$')
    plt.yscale('log')
    plt.title('Continuous-Time Error Estimation')
    plt.legend(loc='upper right')
    plt.show()



if __name__ == "__main__":

    # Gradient Methods in D or C
    Models(Dis_M=True,Con_M=True)

    # Plot the figures in D
    # Gradient_Plot_D()

    # # Plot the figures in C
    Gradient_Plot_C()

    # Plot the error
    Error_Plot_D()

    # Plot the error
    Error_Plot_C()

# Function for Gradient_Descent

