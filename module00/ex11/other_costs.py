import numpy as np
import sys
sys.path.append("../ex02")
from TinyStatistician import TinyStatistician


def mse_(y, y_hat):
    """
    Description:
        Calculate de MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.ndarray, a vector of dimendion m * 1.
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
        mse: has to be float.
        None if there is a matching dimendion problem.
    Raises:
        This function should not raise any Exceptions.
    """
    return ((y_hat - y).dot(y_hat - y) / len(y))

def rmse_(y, y_hat):
    """
    sklearn -> sqrt(mea_squared_error)
    Description:
        Calculate de RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.ndarray, a vector of dimendion m * 1.
        y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    Returns:
        rmse: has to be float.
        None if there is a matching dimendion problem.
    Raises:
        This function should not raise any Exceptions.
    """
    return (((y- y_hat).dot(y - y_hat) / len(y)) ** 0.5)

def mae_(y, y_hat):
    """
    sklearn -> mean_absolute_error
    """
    return (sum(abs(y_hat - y)) / len(y))

def r2score_(y, y_hat):
    divisor = (y_hat - np.mean(y)).dot(y_hat - np.mean(y))
    dividend = (y_hat - y).dot(y_hat - y)
    return (1 - dividend / divisor)


if __name__ == "__main__":
    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])
    print(mse_(x, y))
    print(rmse_(x, y))
    print(mae_(x, y))
    print(r2score_(x, y))