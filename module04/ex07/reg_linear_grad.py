import numpy as np
from prediction import simple_predict

def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray, with two
    ,for-loop. The three arrays must have compatible dimensions.
    Args:
	y: has to be a numpy.ndarray, a vector of dimension m * 1.
	x: has to be a numpy.ndarray, a matrix of dimesion m * n.
	theta: has to be a numpy.ndarray, a vector of dimension n * 1.
	lambda_: has to be a float.
	Returns:
	A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula for all
	,j.
	None if y, x, or theta are empty numpy.ndarray.
	None if y, x or theta does not share compatibles dimensions.
	Raises:
	This function should not raise any Exception.
	"""
    reg = 0
    for i in range(1, theta.shape[0]):
        reg += theta[i] * theta[i]
    grad = np.zeros((theta.shape[0], 1))
    y_hat = simple_predict(x, theta)
    r = y_hat - y
    print(r)
    return grad

def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray, without any
	,→ for-loop. The three arrays must have compatible dimensions.
	Args:
	y: has to be a numpy.ndarray, a vector of dimension m * 1.
	x: has to be a numpy.ndarray, a matrix of dimesion m * n.
	theta: has to be a numpy.ndarray, a vector of dimension n * 1.
	lambda_: has to be a float.
	Returns:
	A numpy.ndarray, a vector of dimension n * 1, containing the results of the formula for all
	,→ j.
	None if y, x, or theta are empty numpy.ndarray.
	None if y, x or theta does not share compatibles dimensions.
	Raises:
	This function should not raise any Exception.
    """
    X_prime = np.c_[np.ones(x.shape[0]), x]
    y_hat = X_prime.dot(theta)
    theta2 = theta.copy()
    theta2[0] = 0
    reg = theta2 * lambda_
    J = (X_prime.T.dot(y_hat - y) + reg) / len(x)
    return J

if __name__ == "__main__":
    x = np.array([
    [ -6, -7, -9],
	[ 13, -2, 14],
	[ -7, 14, -1],
	[ -8, -4, 6],
	[ -5, -9, 6],
	[ 1, -5, 11],
	[ 9, -11, 8]])
    y = np.array([[2],[14],[-13],[5],[12],[4],[-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])
  #  print(reg_linear_grad(y, x, theta, 1))
    print(vec_reg_linear_grad(y, x, theta, 1))
    print(theta)
  #  print(reg_linear_grad(y, x, theta, 0.5))
    print(vec_reg_linear_grad(y, x, theta, 0.5))
  #  print(reg_linear_grad(y, x, theta, 0.0))
    print(vec_reg_linear_grad(y, x, theta, 0.0))