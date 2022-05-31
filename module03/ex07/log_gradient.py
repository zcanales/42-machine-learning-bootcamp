import math as mt
import numpy as np
import sys
sys.path.append("../ex04")
from log_pred import logistic_predict_ as log_pred_

def log_gradient(x, y, theta):
    """
    Compute a gradient vector from three non-empty numpy.array, with a for-loopr.
    The three arrays must have compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a matrix o dimendion m * n.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarrat, a vector (n + 1) * 1.
    Returns:
    The gradient as a numpy.ndarray, a vector of dimensions n * 1, containing the result of the formula for all j.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    y_hat = log_pred_(x, theta)
    X_prime = np.c_[np.ones(x.shape[0]), x]
    J = []
    print(X_prime.shape)
    #Convert y vector to matriix
    y = y.reshape(len(y), 1)
    y_hat = y.reshape(len(y_hat), 1)
    for i in range(X_prime.shape[1]):
        summ = 0
        for j in range(X_prime.shape[0]):
            summ += X_prime[j][i] * (y_hat[j][0] - y[j][0])
        J.append(summ / x.shape[0])    
    return np.array(J)

    #Other way
    y_hat = log_pred_(x, theta).reshape(-1, 1)
    X_prime = np.c_[np.ones(x.shape[0]), x].T
    J = []
    y = y.reshape(len(y), 1)
    y_hat = y.reshape(len(y_hat), 1) 
    for i in range(X_prime.shape[0]):
        summ = 0
        for j in range(X_prime.shape[1]):
            summ += X_prime[i][j] * (y_hat[j][0] - y[j][0])
        J.append(summ / x.shape[0])    
    return np.array(J) 
 

if __name__ == "__main__":
    y1 = np.array([1])
    x1 = np.array([4])
    theta1 = np.array([2, 0.5])
    print(log_gradient(x1, y1, theta1))

    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([2, 0.5])
    print(log_gradient(x2, y2, theta2))

    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(log_gradient(x3, y3, theta3))
