import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
sys.path.append("../ex06")
sys.path.append("../ex06")
from my_linear_regression import MyLinearRegression as MyLR

def print_costfn(t0, y):
    for i in np.linspace(t0 -10, t0 +50, 3000):
        linear_model3 = MyLR(np.array([[-10], [i]]))
        Y_model3 = linear_model3.predict_(Xpill)
        plt.plot(linear_model3.thetas[1], linear_model3.cost_(y, Y_model3), 'gs')

if __name__ == "__main__":
    data = pd.read_csv("are_blue_pills_magics.csv")
    Xpill = np.array(data["Micrograms"]).reshape(-1,1)
    Yscore = np.array(data["Score"]).reshape(-1,1)
    linear_model1 = MyLR(np.array([[89.0], [-8]]))
    linear_model2 = MyLR(np.array([[89.0], [-6]]))
    Y_model1 = linear_model1.predict_(Xpill)
    Y_model2 = linear_model2.predict_(Xpill)


    plt.plot(Xpill, Yscore, 'bo', label = 'Strue (pills)' ) 
    plt.plot(Xpill, Y_model1, color='g', linestyle='--', label = 'Spredict (pills', marker='s')
    plt.xlabel("Quantity of blue pill (in micrograms)")
    plt.ylabel("Space driving score")
    plt.grid(True)
    plt.legend()
    plt.show()

    print(linear_model1.mse_(Xpill, Yscore))
    print(mean_squared_error(Yscore, Y_model1))
    print(linear_model2.mse_(Xpill, Yscore))
    print(mean_squared_error(Yscore, Y_model2))

    theta = linear_model1.fit_(Xpill, Yscore)
    theta0 = int(theta[0])
    theta1 = int(theta[1])
    for j in range(theta0 - 5, theta0 + 5):
        J_elem = []
        for i in range(theta1 - 5, theta1 + 5):
            linear_model3 = MyLR(np.array([[j], [i]]))
            J_elem.append((linear_model3.cost_(Xpill, Yscore)))
        plt.plot(np.arange(theta1 - 5 , theta1 + 5), np.array(J_elem),'bo--', label = f'J(theta0 = {j})' )

    
    plt.ylabel("Cost function J(O0, O1")
    plt.grid(True)
    plt.xlabel("theta 1")
    plt.legend()
    plt.show()
