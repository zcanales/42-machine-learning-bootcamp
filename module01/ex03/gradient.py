import numpy as np
import sys
sys.path.append("../../module00/ex07")
from cost import predict_

def simple_gradient(x, y, theta):
    """Computes the vector of from three non-empty numpy.ndarray, without any for-loop.
        The three arrays must have compatibe dimensions.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        The dragient as numpy.ndarray, a ector of dimension 2 * 1.
        None if x or theta are empty numpy.ndarray.
        None if x, y or theta dimensions are not compatible.
    Raises:
        This function should not raise any Exception.
    """
    J_gradient = np.zeros((2,1))
    print(predict_(x, theta))
    J_gradient[0]  = theta[0] - sum(predict_(x, theta) - y) / len(x)
    J_gradient[1]  = theta[1] - ((predict_(x, theta) - y).dot(x)) / len(x)
    return J_gradient

if __name__ == "__main__":
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
    theta1 = np.array([2, 0.7])
    print(simple_gradient(x, y, theta1))
    theta2 = np.array([1, -0.4])
    print(simple_gradient(x, y, theta2))