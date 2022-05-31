import numpy as np
import pandas as pd
import sys
sys.path.append("../ex07")
sys.path.append("../ex10")
sys.path.append("../ex13")
sys.path.append("../ex11")
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR
from polynomial_train import polynomial_train
from data_spliter import *
import matplotlib.pyplot as plt

def data_spliter2(x, y, proportion):
    indexes = np.arange(y.shape[0])
    np.random.shuffle(indexes)
    return(np.split(x[indexes], [int(x.shape[0] * proportion)]),\
        np.split(y[indexes], [int(y.shape[0] * proportion)]))

if __name__ == "__main__":
    data = pd.read_csv("are_blue_pills_magics.csv")
    x = data[['Micrograms']].values
    y = data[['Score']].values

    #Get Train and Test data
    Xtrain, Xtest, Ytrain, Ytest = data_spliter(x, y, 0.8)
    print(f"Xtrain {Xtrain}")
    print(f"Xtest {Xtest}")
    print(f"Ytrain {Ytrain}")
    print(f"Ytest {Ytest}")


    #Polinomial_feature
    #Linear Regresion
    range_power = 6
    cost_array = []
    for i in range(1,range_power):
        cost_array.append(polynomial_train(Xtrain, Ytrain, i, alpha=5e-10, max_iter=100))
    plt.plot(x, y, 'k', label = 'REAL')
    plt.show()


    #Plot cost
    plt.title("Cost vs order")
    plt.ylabel("Cost")
    plt.xlabel("Order")
    plt.plot(np.arange(1, range_power), np.array(cost_array), 'go', label = 'COST')
    plt.show()  

    #Train vs Test
    print(np.array(cost_array).min())


