import numpy as np
import matplotlib.pyplot as plt

def data_spliter(x, y, proportion):
    """Shuffls and splits the dataset (given by and y ) into a training and a test set,
        while respecting the given porportion of examples to be kept in the training set.
        Args:
            x: has to be an numpy.ndarray, a matrix of dimenion m * n.
            y: ha to be an numpy.ndarray, a vector of dimension m * 1.
            proportion. has to be a ploat, the proportion of the dataset htat will be assigned to the training set.
        Returns:
            (x_train, x_test, y_train, y_test) as a tuple of numpy.ndarray.
            None if x or y is an empty numpy.ndarray.
            None if x and y do not share compatible dimensions.
        Raises:
            This function should not raise any Exception.
        """
    c = np.array(list(zip(x,y)),  dtype=object)
    np.random.shuffle(c)
    array2 = np.split(c, [int(x.shape[0] * proportion)])
#   Xtrain array[0], first colum[:, 0]
    Xtrain = array2[0][:, 0]
#   Xtest array[1], first colum[:, 0]
    Xtest = array2[1][:, 0]
    #Ytrain array[0], second colum[:, -1]
    Ytrain = array2[0][:, -1]
    #Ytest array[1], second colum[:, -1]
    Ytest = array2[1][:, -1]
    return Xtrain, Xtest, Ytrain, Ytest



if __name__ == "__main__":
    y = np.array([0, 1, 0, 1, 0])

    #TEST 1
    x1 = np.array([1, 42, 300, 10, 59])
    Xtrain, Xtest, Ytrain, Ytest = data_spliter(x1, y, 0.8)
    print(f"Xtrain {Xtrain}")
    print(f"Xtest {Xtest}")
    print(f"Ytrain {Ytrain}")
    print(f"Ytest {Ytest}")

    #TEST 2
    print(data_spliter(x1, y, 0.5))

    #TEST 
    x2 = np.array([[1, 42], [300, 10], [59, 1], [300, 59], [10, 24]])
    Xtrain, Xtest, Ytrain, Ytest = (data_spliter(x2, y, 0.8))
    print(f"Xtrain {Xtrain}")
    print(f"Xtest {Xtest}")
    print(f"Ytrain {Ytrain}")
    print(f"Ytest {Ytest}")
