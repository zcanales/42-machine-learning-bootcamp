import numpy as np

class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.00001, max_iter=100000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
    
    def add_intercept(self, x):
        if x.ndim == 1:
            x = np.insert(x, [0], 1, axis = 0)
        else: 
            x = np.insert(x, [0], 1, axis = 1)
        return(x)

    def predict_(self, x):
        X_prime = self.add_intercept(x)
        y_hat = X_prime.dot(self.thetas)
        return y_hat

    def mse_(self, x, y):
        y_hat = self.predict_(x)
        return (float((y_hat - y).T.dot(y_hat - y) / len(y)))

    def fit_(self, x, y, alpha=0.00001, max_iter=100000):
        self.alpha = alpha
        self.max_iter = max_iter
        i = 0
        while i < self.max_iter:
            for n in range(len(self.thetas)):
                self.thetas = self.thetas - self.alpha * ((self.gradient_(x, y)))
            i += 1
        return self.thetas     

    def gradient_(self, x, y):
        X_prime = self.add_intercept(x)
        y_hat = self.predict_(x)
        J = X_prime.T.dot(y_hat - y) / len(x)
        return J
    
    def cost2_(self, y, y_hat):
        return(np.transpose((y - y_hat)).dot(y - y_hat) /(len(y) * 2))

    def cost_elem_(self, x, y):
        ret = []
        y_hat = self.predict_(x)
        y = np.squeeze(y)
        for i in range(len(y)):
            ret.append((1/(2*len(y))) * ((y_hat[i] - y[i])**2))
        return(np.array(ret))

    
    def cost_(self, x, y):
        y_hat = self.predict_(x)
        c = (y - y_hat).T.dot(y - y_hat) / len(x) / 2
        return (c)
    def mse_(self, x, y):
        m = len(x)
        y_hat = self.predict_(x) 
        y = np.squeeze(y)
        c = (((y_hat - y) ** 2) / (m))
        return sum(c)


if __name__ == "__main__":
    x = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    y = np.array([[23.], [48.], [218.]])
    MyLR = MyLinearRegression([[1.], [1.], [1.], [1.], [1.]])

    print(MyLR.predict_(x))
    print(MyLR.cost_elem_(x, y))
    print(MyLR.cost_(x, y))
    print(MyLR.fit_(x, y, alpha=1.6e-4, max_iter=200000))
    print(MyLR.predict_(x))
    print(MyLR.cost_elem_(x, y))
    print(MyLR.cost_(x, y))