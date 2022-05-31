import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
class MyLogisticRegression():
    """
    Description:
        My personnal logistic regression class to fit like a boss.
    """
    def __init__(self, theta, alpha=0.001, max_iter=10000):
        self.theta = theta
        self.alpha = alpha
        self.max_iter = max_iter
    
    def add_intercept(self, x):
        return np.c_[np.ones(x.shape[0]), x]
        if x.ndim == 1:
            x = np.insert(x, [0], 1, axis = 0)
        else: 
            x = np.insert(x, [0], 1, axis = 1)
        return (x)
    
    def predict_(self, x):
        X_prime = np.c_[np.ones((len(x), 1)), x]
        y_hat = 1 / (1 + np.exp(- (X_prime.dot(self.theta))))
        return y_hat

    def gradient_(self, x, y):
        #y_hat and y -> same dimension. y_hat.reshape!!!
        y_hat = self.predict_(x).reshape(-1, 1)

        X_prime = np.c_[np.ones(x.shape[0]), x]
        J = X_prime.T.dot(y_hat - y) / len(x)
        return J

    def cost_(self, x, y):
        eps = float(1e-15)
        ones = np.ones(y.shape[0]).reshape(-1, 1)
        y_hat = self.predict_(x).reshape(-1, 1)
        y_resta = []
        y_ones = []
        for i in range(y_hat.shape[0]):
            y_resta.append(1 - y_hat[i] + eps)
            y_ones.append(1- y[i])
        y_resta = np.array(y_resta).reshape(-1, 1)
        #Other way
     #   y_ones = np.array(y_ones).reshape(-1, 1)
        y_ones = np.subtract(ones, y)
        cost1 = y.T.dot(np.log(y_hat + eps))
        cost0 = y_ones.T.dot(np.log(y_resta))    

        J = (-1 / len(y)) * (cost1 + cost0)
        return J

    def fit_(self, x, y):
        i = 0
        n_cycles =  self.max_iter
        while n_cycles != 0:
            for n in range(len(self.theta)):
                self.theta[n] = self.theta[n]- self.alpha * np.sum((self.gradient_(x,y)[n,:]))
            n_cycles -= 1
        return(self.theta)
        y = np.squeeze(y)
        while i < self.max_iter:
            self.theta = self.theta - self.alpha * (self.gradient_(x, y))
            i += 1
        return self.theta
    def score(self, X, y):
        n_samples = X.shape[0]
        y_pred = self.predict_(X)
        score = 1 - 1/n_samples * (np.abs(y - y_pred)).sum()
        return score


if __name__ == "__main__":  
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])

    mylr = MyLogisticRegression([2, 0.5, 7.1, -4.3, 2.09])
    print(mylr.predict_(X))
    print(mylr.cost_(X, Y))
    print(mylr.fit_(X, Y))
    print(mylr.predict_(X))
    print(mylr.cost_(X, Y))
    
    plt.plot(X, Y, 'bo', label = 'Sell price' ) 
    plt.plot(X, mylr.predict_(X) , 'co', label = 'Predicted sell price')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    logisticRegr = LogisticRegression(max_iter=1000)
    print(Y.shape)
    print(Y.ravel().shape)
    logisticRegr.fit(X_train, y_train.ravel())
    print(logisticRegr.predict(X))
    print(logisticRegr.score(X, Y))
    
    plt.plot(X, logisticRegr.predict(X) , 'ro', label = 'Predicted sell price python library')
    
    plt.legend()
    plt.show()