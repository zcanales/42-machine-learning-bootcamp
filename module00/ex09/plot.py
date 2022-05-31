import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../ex04")
sys.path.append("../ex08")
from prediction2 import predict_
from vec_cost import cost_

def plot_with_cost(x, y,theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    y_hat = predict_(x, theta)
    plt.plot(x, y, 'bo', label = 'real' ) 
    plt.plot(x, y_hat, 'r', label = 'predicted')
    J_elem = cost_(y, y_hat)
    plt.vlines(x, y, y_hat, label = "cost", linestyles="dashed", colors="k")
    plt.suptitle(f'Cost = {J_elem}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
    theta1 = np.array([18, -1])
    plot_with_cost(x, y, theta1)
    theta2 = np.array([14, 0])
    plot_with_cost(x, y, theta2)
