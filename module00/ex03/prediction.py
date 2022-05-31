def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    #Function in precition (simple_predictio) 
    y = []
    for i in x:
        y.append(theta[0] + theta[1] * i)
    return (y)

if __name__ == "__main__":
    theta1 = [5, 0]
    print(simple_predict(range(1, 6), theta1))
    theta2 = [0, 1]
    print(simple_predict(range(1, 6), theta2))
    theta3 = [5, 3]
    print(simple_predict(range(1, 6), theta3)) 
    theta4 = [-3, 1]
    print(simple_predict(range(1, 6), theta4))