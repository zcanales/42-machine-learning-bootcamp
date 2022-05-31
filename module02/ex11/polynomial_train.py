import numpy as np
import pandas as pd
import sys
sys.path.append("../ex07")
sys.path.append("../ex10")
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt

def polynomial_train(x, y, power, alpha, max_iter):
    #Power x
    x_poly = add_polynomial_features(x, power)
    
    #Create MyLR
    theta = np.c_[80, np.ones((1, power + 1))].reshape(1 , -1)
    mylr = MyLR(theta.squeeze(),  alpha, max_iter)
   
   #Cost
    print(mylr.cost_(x_poly, y))
   
   #Fit thetas
    mylr.fit_(x_poly, y)
    cost_after = mylr.cost_(x_poly, y)
    
    #Predict y_hat
    y_hat = mylr.predict_(x_poly)
    
    #Plot
    color = ['bo', 'go', 'ro', 'co', 'mo', 'yo']
    plt.plot(x, y_hat, color[(power - 1) % 6], label = f'power{power}' )
    plt.legend()

    return cost_after ** (1/power)

if __name__ == "__main__":
    data = pd.read_csv("are_blue_pills_magics.csv")
    x = np.array(data[['Micrograms']]).reshape(-1, 1)
    y = np.array(data[['Score']]).reshape(-1,1)
    cost_array = []
    for i in range(1, 6):
        cost_array.append(polynomial_train(x, y, i))
    plt.plot(x, y, 'k', label = 'REAL')
    plt.show()

    #Plot cost
    plt.title("Cost vs order")
    plt.ylabel("Cost")
    plt.xlabel("Order")
    plt.plot(np.arange(1, 6), np.array(cost_array), 'go', label = 'COST')
    plt.show()



    x = np.arange(1, 11)
    y = np.array([[1.39270298], [3.88237651], [4.37726357], [4.63389049], [7.79814439], [6.41177462], [8.63429886], [8.19939795], [10.37567392], [10.6823822]])



