import numpy as np

def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns to every power in the range of 
    1 up to the power given in argument.
    Args:
    x: has to be an numpy.ndarray, a matrix of dimension m * n.
    power: has to be an int, the power up to which the columns of matrix x are going to be
    raised.
    Returns:
    The matrix of polynomial features as a numpy.ndarray, of dimension m * (np), containg the 
    polynomial feature values for all training examples.
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    poly = np.zeros((x.shape[0], power * x.shape[1]))
    #1 Way
    p = 1
    for k in range(poly.shape[1]):
        form = (1 + (-1) ** (k + 1)) / 2
        poly[:, k] = x[:, int(form)] ** (int(p))
        p += 0.5
    return poly
    #2 Way
    col = 0
    for i in range(power):
        for j in range(x.shape[1]):
            poly[:, col] = x[:, j] ** (i + 1)
            col += 1


if __name__ == "__main__":
    x = np.arange(1,11).reshape(5, 2)
    print(x)
    print(add_polynomial_features(x, 3))
    print(add_polynomial_features(x, 4))