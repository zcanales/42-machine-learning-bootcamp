import math as mt
import numpy as np
import sys
sys.path.append("../ex04")
from log_pred import logistic_predict_ as log_pred_

def vec_log_loss_(y, y_hat, eps=1e-15):
    """
    Compute the logistic loss value.
    Args:
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
    eps: epsilon (default=1e-15)
    Returns:
    The logistic loss value as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    ones = np.ones(y.shape[0]).reshape(-1, 1)
    y_resta = []
    for i in range(y_hat.shape[0]):
        y_resta.append(1 - y_hat[i] - eps)
    y_resta = np.array(y_resta).reshape(-1, 1)
    y_ones = np.subtract(ones, y)

    cost0 = y_ones.T.dot(np.log(y_resta))    
    cost1 = y.T.dot(np.log(y_hat + eps))
    
    J = (-1 / len(y)) * (cost1 + cost0)
    return J

if __name__ == "__main__":
    y1 = np.array([1])
    x1 = np.array([4])
    theta1 = np.array([2, 0.5])
    y_hat1 = log_pred_(x1, theta1)
    print(vec_log_loss_(y1, y_hat1))

    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([2, 0.5])
    y_hat2 = log_pred_(x2, theta2)
    print(vec_log_loss_(y2, y_hat2))

    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    y_hat3 = log_pred_(x3, theta3)

    print(vec_log_loss_(y3, y_hat3))