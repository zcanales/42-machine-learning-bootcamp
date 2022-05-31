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

#Classifier 1
X1 = []
y1 = []
for i in range(len(Xtrain)):
    if Ytrain[i] == 0 or Ytrain[i] == 1:
        X1.append(Xtrain[i])
        y1.append(Ytrain[i])
X1 = np.array(X1)
y1 = np.array(y1)
plt.scatter(X1[:, 0], X1[:, 1], c=y1, cmap='plasma')
plt.show()
print(X1[:, 0])
print(X1[:, 1])

X2 = []
y2 = []
for i in range(len(Xtrain)):
    if Ytrain[i] == 0 :
        X2.append(Xtrain[i])
        y2.append(0)
    if Ytrain[i] == 2:
        X2.append(Xtrain[i])
        y2.append(1)
X2 = np.array(X2)
y2 = np.array(y2)
plt.scatter(X2[:, 0], X2[:, 1], c=y2, cmap='plasma')
plt.show()
