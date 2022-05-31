import matplotlib.pyplot as plt
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
    if x.shape[1] != 1:
        return
    if not any(x):
        return 
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
    y_hat = add_intercept(x)
    if y_hat is None:
        return
    if not isinstance(theta, np.ndarray) or theta.shape != (2, 1):
        return None 
    return np.dot(y_hat, theta)

def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """
    y_hat = predict_(x, theta)
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    axes.scatter(x, y, label = 'raw', c='#ff0000')
    axes.plot(x, y_hat, 'r--', label = 'prediction', c='#4287f5')
    #Axis(X) and Axis (Y)
    plt.suptitle('Plotting')
    plt.legend()
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()
    #Otra manera -> plt.plot(x, y, 'r--', x, y_hat, 'bs')


if __name__ == "__main__":
    x = np.arange(1,6).reshape(-1, 1)
    y = np.array([[5],[3],[2.3],[2.4],[5]])
    
    theta1 = np.array([[5],[0]])
    plot(x, y, theta1)
    theta1 = np.array([[0],[1]])
    plot(x, y, theta1)
    theta1 = np.array([[5],[3]])
    plot(x, y, theta1)
    theta1 = np.array([[-3],[1]])
    plot(x, y, theta1)
