import numpy as np

class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
    def mse_(self, x, y):
        y_hat = self.predict_(x)
        return (float((y_hat - y).T.dot(y_hat - y) / len(y)))

    def fit_(self, x, y):
        i = 0
        while i < self.max_iter:
            i += 1
            self.thetas = self.thetas - self.alpha * self.gradient_(x, y)
        return self.thetas     

    def gradient_(self, x, y):
        y_hat = self.predict_(x)
        X_prime = np.c_[np.ones(x.shape[0]), x]
        return ((np.transpose(X_prime).dot(y_hat - y)) / len(y))

    def predict_(self, x):
         X_prime = np.c_[np.ones(x.shape[0]), x]
        return (np.dot(X_prime, self.thetas))

    def cost_elem_(self, x, y):
        y_hat = self.predict_(x)
        return (((y_hat - y) ** 2) / len(y) / 2)
    
    def cost_(self, x, y):
        y_hat = self.predict_(x)
        return (float(((y_hat - y).T.dot(y_hat - y)) / len(x) / 2))

    
if __name__ == "__main__":
    MyLR = MyLinearRegression([2, 0.7])
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
    
    print(MyLR.predict_(x))
    print(MyLR.cost_elem_(MyLR.predict_(x), y))
    print(MyLR.cost_(MyLR.predict_(x), y))
    
    
    MyLR2 = MyLinearRegression([0, 0])
    thetas = MyLR2.fit_(x, y)
    print(thetas)
    print(MyLR2.predict_(x))
    print(MyLR2.cost_elem_(MyLR2.predict_(x), y))
    print(MyLR2.cost_(MyLR2.predict_(x), y))