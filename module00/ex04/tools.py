import numpy as np
#from tools import add_intercept

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
#    return (add_intercept(x))
    

if __name__ == "__main__":
    print(add_intercept(np.array([[1], [2], [3]])))
    print(add_intercept(np.array([[0], [0], [0]])))
    print(add_intercept(np.array([[0, 2, 3]])))
    print(add_intercept([0, 2, 3]))