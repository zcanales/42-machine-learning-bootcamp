import pandas as pd
import numpy as np
import sys
sys.path.append("../ex03")
sys.path.append("../../module01/ex06")
sys.path.append("../ex09")
from my_linear_regression import  MyLinearRegression
from polynomial_model_extended import add_polynomial_features
from ridge import MyRidge 
import matplotlib.pyplot as plt

def mse_(y, y_hat):
    return ((y_hat - y).T.dot(y_hat - y) / len(y))

#Split
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model


#0. Get data
data = pd.read_csv("spacecraft_data.csv")
x = np.array(data[['Age','Thrust_power','Terameters']])
y = np.array(data[['Sell_price']])

#1. Slipt dataset int traing and test set.
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3, random_state=0)

#2. Polinomia grade 3
Xtrain_poly3 = add_polynomial_features(Xtrain, 3)
Xtest_poly3 = add_polynomial_features(Xtest, 3)

model = {}
mse = {}
thetas = np.ones(4)
for lambda_ in np.arange(0.1, 1, 0.1):
    lambda_ = round(lambda_, 1)
    model[lambda_] = MyRidge(thetas, alpha=5e-5, max_iter=500000, lambda_=lambda_)
    model[lambda_].fit_(Xtrain, Ytrain)
    print("Cost")
    mse[lambda_] = model[lambda_].cost_(x_test, y_test, lambda_)
    print("MSE " + str(lambda_) + " : ", mse[lambda_])
exit()
#3.
from sklearn.linear_model import Ridge
thetas = np.ones(Xtrain_poly3.shape[1] + 1).reshape((-1,1))
rang = np.arange(0.1, 1, 0.1)
mtlr = MyLinearRegression(thetas, alpha = 5e-25, max_iter = 100)
res = []
mse_result = []
for l in rang:
    myLR_Ridg = Ridge(alpha=100, max_iter=2000,)
    myLR_Ridg.fit(Xtrain_poly3, Ytrain)
    print(mse_(Ytrain, myLR_Ridg.predict(Xtrain_poly3)))

