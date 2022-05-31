import numpy as np

class MyRidge():
    """
    Description:
    My personnal ridge regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        self.thetas = thetas
        self.alpha = alpha
        self.max_iter = max_iter
        self.lambda_ = lambda_

    def get_params_(self):
        return ({'thetas': self.thetas, 'alpha': self.alpha, 'max_iter': self.max_iter, 'lambda_':self.lambda_})
    
    def set_params_(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        self.thetas = thetas
        self.alpha = alpha
        self.max_iter = max_iter
        self.lambda_ = lambda_

    def predict_(self, x):
        X_prime = np.c_[np.ones(x.shape[0]) , x]
        y_hat = X_prime.dot(self.thetas)
        return y_hat

    def l2(self, theta):
        theta2 = self.thetas.copy()
        theta2[0] = 0
        ret = theta2.T.dot(theta2)
        return ret

    def reg_cost_(self, y, y_hat):
        ret = 0
        ret = self.thetas.T.dot(self.thetas)
        cost = (y_hat - y).T.dot(y_hat - y) 
        return ((cost + (self.lambda_ * ret)) / 2 / y.shape[0])

    def reg_linear_grad(self, x, y):
        X_prime = np.c_[np.ones(x.shape[0]), x]
        y_hat = self.predict_(x)
        theta2 = self.thetas.copy()
        theta2[0] = 0
        reg = 0
        reg = theta2 * self.lambda_
        J = (X_prime.T.dot(y_hat - y) + reg) / len(x)
        return J    
    
    def fit_(self, x, y):
        i = 0
        while i < self.max_iter:
            for n in range(len(self.thetas)):
                self.thetas = self.thetas - self.alpha * ((self.reg_linear_grad(x, y)))
            i += 1
        return self.thetas
    
    def confusion_matrix_(self, y_true, y_hat, labels=None, df_option=False):
        if labels == None:
            if len(np.unique(y_true)) > len(np.unique(y_hat)):
                labels = np.unique(y_true)
            else:
                labels = np.unique(y_hat)
            labels = labels.tolist()
        ret = np.zeros((len(labels),len(labels)))
        for i in range(y.shape[0]):
            if y_true[i] in labels and y_hat[i] in labels:
                ret[labels.index(y_true[i]) ][labels.index(y_hat[i])] += 1
        if df_option == True:
            ret = pd.DataFrame(ret, labels, labels)
        return(ret)
    
    def accuracy_score_(self, y, y_hat):
        ac = 0
        for yi, y_hati in zip(y, y_hat):
            if yi == y_hati:
                ac += 1
        return ac / len(y)
    def f1_score_(self, y, y_hat, pos_label=1):
        ps = precision_score_(y, y_hat, pos_label)
        rc = recall_score_(y, y_hat, pos_label) 
        return ((2 * ps * rc) / (ps + rc))  


if __name__ == "__main__":
    x = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    y = np.array([[23.], [48.], [218.]])
    MyLR = MyRidge(np.array([[1.], [1.], [1.], [1.], [1.]]), alpha=1.6e-6, max_iter =20000)
    from sklearn.linear_model import Ridge
    clf = Ridge(alpha=1.6e-6, max_iter=2000)
    
    print(MyLR.predict_(x))
    print(MyLR.fit_(x, y))
    print(MyLR.predict_(x))
   
    print(clf.fit(x, y))
    print(clf.predict(x))
    print(clf.get_params())
    print(clf.score(x, y))
