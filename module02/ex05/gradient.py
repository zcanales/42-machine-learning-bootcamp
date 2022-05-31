import numpy as np
import sys
sys.path.append("../ex03")
from prediction import simple_predict

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, without any for loop. 
        The three arrays must have compatible dimensions.
    Args:
        x: has to be a numpy.ndarray, a matrix of dimension m * n.
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        theta: has to be a numpy.ndarray, a (n + 1) * 1 vector.
    Returns:
        The gradient as a numpy.ndarray, a vector of dimension n * 1, containg the result of formula for all j.
        None if x, y, or theta is an empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    X_prime = np.c_[np.ones(x.shape[0]), x]
    y_hat = simple_predict(x, theta)
    J = X_prime.T.dot(y_hat - y) / len(x)
    return J


if __name__ == "__main__":
    x = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])
    y = np.array([2, 14, -13, 5, 12, 4, -19])
    theta1 = np.array([1, 3, 0.5, -6])
    print(gradient(x, y, theta1))
    
    theta2 = np.array([0.85, 0, 0, 0])
    print(gradient(x, y, theta2))