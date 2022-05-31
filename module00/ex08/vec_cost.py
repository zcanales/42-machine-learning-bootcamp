import numpy as np

def cost_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
        None if y or y_hat is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or y.shape != y_hat.shape:
        return None
    J_elem = np.dot(y_hat - y , y_hat- y) / (2 * y.shape[0])
    #Other way
    #J_elem = (y - y_hat).dot(y - y_hat) /(2 * len(y))
    return J_elem

if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    print(cost_(X, Y))
    print(cost_(X, X))