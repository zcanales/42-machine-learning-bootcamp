import numpy as np
import sys
sys.path.append("../ex03")
sys.path.append("../ex04")
sys.path.append("../ex05")
from prediction import simple_predict
from gradient import gradient
from cost import cost_

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * n: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension (n + 1) * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension (n + 1 * 1).
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    i = 0
    while i < max_iter:
        i += 1
        theta = theta - alpha * gradient(x, y, theta)
    return theta 

if __name__ == "__main__":
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])
    #theta more than 0.0005 -> nan
    theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
    print(theta2)

    y_hat = simple_predict(x, theta2)
    print(y_hat)

    cost = cost_(y, y_hat)
    print(cost)

