import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
    x: has to be an numpy.ndarray, a vector
    Returns:
    The sigmoid value as a numpy.ndarray.
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    sig = np.array(1 / (1 + np.exp(-x)))
    return sig


if __name__ == "__main__":
    x = np.array(-4)
    print(sigmoid_(x))
    print(sigmoid_(np.array(2)))
    print(sigmoid_(np.array([[-4], [2], [0]])))
    x_plot = np.array([[-30], [-20], [-10], [-8],[-4], [0], [2],[6], [10], [20], [30]])
    y_sig = (sigmoid_(x_plot))
    print(y_sig)
    plt.plot(x_plot, y_sig)
    plt.show()

