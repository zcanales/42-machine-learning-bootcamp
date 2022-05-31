import numpy as np


def simple_predict(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n  + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    X_prime = np.c_[np.ones(x.shape[0]) , x]
    y_hat = X_prime.dot(theta)
    return y_hat

    
if __name__ == "__main__":
    x = np.arange(1,13).reshape((4,3))

    theta1 = np.array([5, 0, 0, 0])
    print(simple_predict(x, theta1))
    
    theta2 = np.array([0, 1, 0, 0])
    print(simple_predict(x, theta2))
    
    theta3 = np.array([-1.5, 0.6, 2.3, 1.98])
    print(simple_predict(x, theta3))
    
    theta4 = np.array([-3, 1, 2, 3.5])
    print(simple_predict(x, theta4))