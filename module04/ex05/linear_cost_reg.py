import numpy as np

def reg_cost_(y, y_hat,theta, lambda_):
    """Computes the regularized cost of a linear regression model from two non-empty,
    numpy.ndarray, without any for loop. The two arrays must have the same dimensions.
	Args:
	y: has to be an numpy.ndarray, a vector of dimension m * 1.
	y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
	theta: has to be a numpy.ndarray, a vector of dimension n * 1.
	lambda_: has to be a float.
	Returns:
	The regularized cost as a float.
	None if y, y_hat, or theta are empty numpy.ndarray.
	None if y and y_hat do not share the same dimensions.
	Raises:
	This function should not raise any Exception.
	"""
    ret = 0
    ret = theta.T.dot(theta)
    cost = (y_hat - y).T.dot(y_hat - y) 
    return ((cost + (lambda_ * ret)) / 2 / y.shape[0])

if __name__ == "__main__":
    y = np.array([2, 14, -13, 5, 12, 4, -19])
    y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20])
    theta = np.array([1, 2.5, 1.5, -0.9])

    print(reg_cost_(y, y_hat, theta, .5))
    print(reg_cost_(y, y_hat, theta, .05))
    print(reg_cost_(y, y_hat, theta, .9))