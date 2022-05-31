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
#    if x.shape[1] != 1:
#        return
 #   if not any(x):
  #      return 
    return (np.c_[np.ones(x.shape[0]), x])
#   return (np.hstack((np.ones(x.shape[0]), 1)), x)
#   return (add_intercept(x))

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
#    if not isinstance(theta, np.ndarray) or theta.shape != (2, 1):
 #       return None 
    return np.dot(x, theta)

def cost_elem_(y, y_hat):
    """
    Description:
    Calculates all the elements (1/2*M)*(y_pred - y)^2 of the cost function.
    Args:
    y: has to be an numpy.ndarray, a vector.
    y_hat: has to be an numpy.ndarray, a vector.
    Returns:
    J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
    None if there is a dimension matching problem between X, Y or theta.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) \
        or not isinstance(y_hat, np.ndarray) or\
        y.shape != y_hat.shape:
        return None
   # J_elem = np.zeros(y.shape)
    #for i in range(y.shape[0]):
     #   J_elem[i] = (0.5 / y.shape[0]) * (y_hat[i] - y[i]) ** 2
    J_elem = []
    for yi_hat, yi in zip(y_hat, y):
        J_elem.append((0.5 / y.shape[0]) * (yi_hat - yi) ** 2)
    return np.array(J_elem)

def cost_(y, y_hat):
    """
    Description:
    Calculates the value of cost function.
    Args:
    y: has to be an numpy.ndarray, a vector.
    y_hat: has to be an numpy.ndarray, a vector.
    Returns:
    J_value : has to be a float.
    None if there is a dimension matching problem between X, Y or theta.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) \
        or not isinstance(y_hat, np.ndarray) or\
        y.shape != y_hat.shape:
        return None
    J_elem = 0
    for i in range(y.shape[0]):
        J_elem += (0.5 / y.shape[0]) * (y_hat[i] - y[i]) ** 2 
    return float(J_elem)


if __name__ == "__main__":
    print("TEST 1")
    x = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.],[4.]])
    y_hat = predict_(x, theta1)
    y = np.array([[2],[7],[12],[17.],[22.]])
    J_elem = cost_elem_(y, y_hat)
    print(J_elem)
    print(cost_(y, y_hat))
    
    print("TEST 2")
    x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80]])
    theta2 = np.array([[0.05],[1.], [1.], [1.]])
    y_hat2 = predict_(x2, theta2)
    y2 = np.array([[19.],[42],[67.],[93.]])
    print(cost_elem_(y2, y_hat2))
    print(cost_(y2, y_hat2))

    print("TEST 3")
    x3 = np.array([0, 15, -9, 7, 12, 3, -21])
    theta3 = np.array([[0.],[1.]])
    y_hat3 = predict_(x3, theta3)
    y3 = np.array([2, 14, -13, 5, 4, -19])
    print(cost_(y3, y_hat3))
    print(cost_(y3, y3))