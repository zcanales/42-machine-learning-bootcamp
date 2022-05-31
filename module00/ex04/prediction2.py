import numpy as np

def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.ndarray x.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    Returns:
    X as a numpy.ndarray, a vector of dimension m * 2.
    None if x is not a numpy.ndarray.
    None if x is a empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    return (np.c_[np.ones(x.shape[0]), x])


def predict_(x, theta):    
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
        y_hat as a numpy.array, a vector of shape m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta shapes are not appropriate.
        None if x or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    x = add_intercept(x)
    if x is None:
        return
    return np.dot(x, theta)

if __name__ == "__main__":
    arr = np.arange(1,6).reshape(-1, 1)
    print(f"arr -> {arr.shape}")
    theta1 = np.array([[5],[0]])
    print(f"shape -> {theta1.shape[1]}")
    print(predict_(arr, theta1))
    theta2 = np.array([[0], [1]])
    print(predict_(arr, theta2))
    theta3 = np.array([[5], [3]])
    print(predict_(arr, theta3)) 
    theta4 = np.array([[-3], [1]])
    print(predict_(arr, theta4))
    