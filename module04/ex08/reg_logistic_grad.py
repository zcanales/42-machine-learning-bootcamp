import numpy as np

def logistic_predict_(x, theta):
    X_prime = np.c_[np.ones((len(x), 1)), x]
    y_hat = 1 / (1 + np.exp(-(X_prime.dot(theta))))
    return y_hat

def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any
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
    y_hat = logistic_predict_(x, theta)
    theta2 = theta.copy()
    theta2[0] = 0
    reg = theta2 * lambda_
    J = (X_prime.T.dot(y_hat - y) + reg) / len(x)
    return J

if __name__ == "__main__":
	x = np.array([[0, 2, 3, 4],
	[2, 4, 5, 5],
	[1, 3, 2, 7]])
	y = np.array([[0], [1], [1]])
	theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

	print(vec_reg_logistic_grad(y, x, theta, 1))

	print(vec_reg_logistic_grad(y, x, theta, 0.5))
	
	print(vec_reg_logistic_grad(y, x, theta, 0.0))