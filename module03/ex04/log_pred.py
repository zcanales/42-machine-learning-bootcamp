import numpy as np
import sys
sys.path.append("../ex00")
from mylinearregression import MyLinearRegression as MyLR

def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * n.
    theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    X_prime = np.c_[np.ones((len(x), 1)), x]
    y_hat = 1 / (1 + np.exp(-(X_prime.dot(theta))))
    return y_hat


if __name__ == "__main__":
    x = np.array([4])
    theta = np.array([[2], [0.5]])
    print(logistic_predict_(x, theta))

    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(logistic_predict_(x2, theta2))

    x3 = np.array([[0, 2, 3, 4], [2, 3, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(logistic_predict_(x3, theta3))
