import pandas as pd
import numpy as np
import sys
sys.path.append("../ex09")
from my_logistic_regression import MyLogisticRegression as MyLR2
import matplotlib.pyplot as plt

#Split
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model


#0. Get data   
X = pd.read_csv("solar_system_census.csv")
Y = pd.read_csv("solar_system_census_planets.csv")

X1 = X.values
X1 = X1[:, 1:]
Y1 = Y.values
Y1 = Y1[:, 1:]

#1. Slipt dataset int traing and test set.
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X1, Y1, test_size=0.3, random_state=0)
#plt.scatter(Xtrain[:, 0], Xtrain[:, 2], c=Ytrain, cmap='plasma')
#plt.show()

#One vs All
mymodel = linear_model.LogisticRegression(max_iter=100000)
nb_theta = 4
precision = 0
recall = 0
for j in range(4):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    Ytrain_onevsall = np.where(Ytrain != j, 0, 1)
    Ytest_0 = np.where(Ytest != j, 0, 1)
    mymodel.fit(Xtrain, Ytrain_onevsall.ravel())
    predicted_output = mymodel.predict(Xtrain)
    print(f"cost: {mymodel.score(Xtest, Ytest)}")
    Ytest_onesvsall = np.where(predicted_output != j, 0, 1)
    print(Ytrain.ravel())
    print(Ytrain_onevsall.ravel())
    print(predicted_output)
    for i in range(predicted_output.shape[0]):
        if Ytrain_onevsall[i][0] == 1 and Ytest_onesvsall[i]== 1:
            tp += 1
        elif Ytrain_onevsall[i][0] == 0 and Ytest_onesvsall[i]== 1:
            fp += 1
        elif Ytrain_onevsall[i][0] == 1 and Ytest_onesvsall[i]== 0:
            fn += 1
        elif Ytrain_onevsall[i][0] == 0 and Ytest_onesvsall[i]== 0:
            tn += 1
    print("For the planet:", (j))
    print("tp:", tp, ", fp:", fp, ", fn:", fn, ", tn:", tn)
    precision += (tp / (tp + fp))
    recall += (tp / (tp + fn))
precision = precision / nb_theta
recall = recall / nb_theta
print("precision:", precision)
print("recall:", recall)
print("f1:", ((2 * (precision * recall)) / (precision + recall)))