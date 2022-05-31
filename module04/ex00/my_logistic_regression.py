import numpy as np

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
        #        self.thetas[n] = self.thetas[n]- self.alpha * np.sum((self.gradient_(x,y)))
            self.theta = self.theta - self.alpha * (self.gradient_(x, y))
            i += 1
        return self.theta

    @staticmethod
    def metrics(y,y_hat,  pos_label=1):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        mtc = np.zeros(4)
        for i in range(y.shape[0]):
            if y[i] == pos_label and y_hat[i] == pos_label:
                tp += 1
                mtc[0] += 1
            elif y[i] != pos_label and y_hat[i] == pos_label: 
                fp += 1
                mtc[1] += 1
            elif y[i] == pos_label and y_hat[i] != pos_label:
                fn += 1
                mtc[2] += 1
            elif y[i] != pos_label and y_hat[i] != pos_label:
                tn += 1
                mtc[3] += 1
        return mtc[0], mtc[1], mtc[2], mtc[3]

    @staticmethod
    def accuracy_score_(y, y_hat):
        ac = 0
        for yi, y_hati in zip(y, y_hat):
            if yi == y_hati:
                ac += 1
        return ac / len(y)

    @staticmethod
    def precision_score_(y, y_hat, pos_label=1):
        tp, fp, fn, tn = metrics(y, y_hat, pos_label)
        return (tp / (tp + fp))

    @staticmethod
    def recall_score_(y, y_hat,  pos_label=1):
        tp, fp, fn, tn = metrics(y, y_hat, pos_label)
        return (tp / (tp + fn))

    @staticmethod
    def f1_score_(y, y_hat, pos_label=1):
        ps = precision_score_(y, y_hat, pos_label)
        rc = recall_score_(y, y_hat, pos_label) 
        return ((2 * ps * rc) / (ps + rc))
    @staticmethod
    def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
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

if __name__ == "__main__":  
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])

    mylr = MyLogisticRegression([2, 0.5, 7.1, -4.3, 2.09])
    print(mylr.predict_(X))
    print(mylr.metrics(Y, mylr.predict_(X)))