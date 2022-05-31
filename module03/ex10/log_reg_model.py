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
#Xtrain, Xtest, Ytrain, Ytest = data_spliter(X1, Y1, 0.8)
plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=Ytrain, cmap='plasma')
plt.show()
#2. Selec your favorit Scapxe Zipcode and generate a new numpy.array to label each citizen according to 
#your new selection criteria.
    #1-> Favorit Planet
    #0-> Other
Ytest_0 = np.where(Ytest == 0, 0, 1)
Ytrain_0 = np.where(Ytrain == 0, 0, 1)

#3. Train logict regression to predict 
mymodel = linear_model.LogisticRegression(max_iter=100000)
mymodel.fit(Xtrain, Ytrain_0.ravel())
predicted_output = mymodel.predict(Xtest)
print(mymodel.score(Xtest, Ytest_0))

mylr = MyLR2(np.array([1, 1]), max_iter=100000)
#mylr.fit_(Xtrain.ravel(), Ytrain.ravel())
#mylr.predict_(Xtrain.ravel())
#print(mylr.score(Xtrain.ravel(), Ytrain.ravel))



#4.Matrix confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Ytest, predicted_output)
print(cm)

#5. Plot
import seaborn as sn
plt.figure(figsize=(5,4))
sn.heatmap(cm, annot=True)
plt.xlabel("Predicted Value")
plt.ylabel("Thruth or Actual Value")
#plt.show()

#6. One vs All
nb_theta = 4
precision = 0
recall = 0
for j in range(4):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    Ytrain_onevsall = np.where(Ytrain == j, 0, 1)
    mymodel.fit(Xtrain, Ytrain_onevsall.ravel())
    predicted_output = mymodel.predict(Xtrain)
    Ytest_onesvsall = np.where(predicted_output == j, 0, 1)
    print(Ytest_onesvsall.shape)
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