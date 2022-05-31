import numpy as np
import math as mt

def reg_log_cost_(y, y_hat, theta, lambda_):
    """Computes the regularized cost of a logistic regression model from two non-empty 
    numpy.ndarray, without any for loop. The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be a numpy.ndarray, a vector of dimension n * 1.
    lambda_: has to be a float.
    Returns:
    The regularized cost as a float.
    None if y, y_hat, or theta is empty numpy.ndarray.
    None if y and y_hat do not share the same dimensions.
    Raises:
    This function should not raise any Exception.
    """
    eps = 1e-15
    theta = theta.copy()
    theta[0] = 0
    reg = theta.T.dot(theta)
    return -(np.sum((y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat)))) * (1 / y.shape[0]) \
		+ (lambda_ * reg / (2 * y.shape[0]))
    ones = np.ones(y.shape[0]).reshape((-1,1))
    y_hat = y_hat.reshape(-1, 1)
    y = y_hat.reshape(-1, 1)
    cost = - (y.T.dot(np.log(y_hat)) + (ones - y).T.dot(np.log(ones - y_hat)))
    return (cost + (lambda_  * reg / (2 * y.shape[0])))


if __name__ == "__main__":
    y = np.array([1, 1, 0, 0, 1, 1, 0])
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01])
    theta = np.array([1, 2.5, 1.5, -0.9])
    print(reg_log_cost_(y, y_hat, theta, .5))
    print(reg_log_cost_(y, y_hat, theta, .05))
    print(reg_log_cost_(y, y_hat, theta, .9))