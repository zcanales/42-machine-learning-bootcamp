import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../ex07")
from mylinearregression import MyLinearRegression as MyLR
from sklearn.linear_model import LinearRegression 

def fit_and_plot(X, Y, theta , s):

    #Gradient Descend
    myLR = MyLR(theta, alpha=2.5e-6, max_iter=10000)
    print(myLR.fit_(X[:,0].reshape(-1,1), Y))
    RMSE = myLR.mse_(X[:, 0].reshape(-1, 1), Y)
    print(RMSE)

    #PLOT
    y_hat = myLR.predict_(X)
    plt.plot(X, Y, 'bo', label = 'Sell price' ) 
    plt.plot(X, y_hat, color='c', label = 'Predicted sell price')
    plt.xlabel("y: sell price (in keuros)")
    plt.ylabel(f"x1: {s}(in years")
    plt.grid(True)
    plt.legend()
    
    plt.show() 

def plot_multivarable(X, Y):

    #Gradient Descend
    ml = MyLR([1.0, 1.0, 1.0, 1.0], alpha=1e-5, max_iter=600000)
    print(ml.mse_(X1, Y))
    print(ml.fit_(X1, Y))
    print(ml.mse_(X1, Y))

    # Create window
    plt.figure(figsize=(20, 7), dpi=70)

    y_hat = ml.predict_(X1)

    # Subplot AGE
    plt.subplot(1, 3, 1)
    plt.title("Age")
    plt.plot(X1[:, 0], Y, 'bo', label = 'Sell price' ) 
    plt.plot(X1[:, 0], y_hat, 'co', label = 'Predicted sell price')
    plt.xlabel('x1: age(in years')
    plt.ylabel('y: sell price(in k euros')

    # Subplot THRUST POWER
    plt.subplot(1, 3, 2)
    plt.title("Thrust_power")
    plt.plot(X1[:, 1], Y, 'go', label = 'Sell price' ) 
    plt.plot(X1[:, 1], y_hat, 'yo', label = 'Predicted sell price')
    plt.xlabel('x1: thrust power (in 10 Km/s')
    plt.ylabel('y: sell price(in k euros')

    # Subplot TERAMETERS
    plt.subplot(1, 3, 3)
    plt.title("Terametersr")
    plt.plot(X1[:, 2], Y, 'ro', label = 'Sell price' ) 
    plt.plot(X1[:, 2], y_hat, 'mo', label = 'Predicted sell price')
    plt.xlabel('x1: terameters (in 10 Km)')
    plt.ylabel('y: sell price(in k euros')

    plt.show()




if __name__ == "__main__":
    data = pd.read_csv("spacecraft_data.csv")
    Y = np.array(data[['Sell_price']])
    
    #Simple variable
    theta = np.array([1000.0, -1.0])
    fit_and_plot(np.array(data[['Age']]), Y, theta, "Age")
    fit_and_plot(np.array(data[['Thrust_power']]), Y, np.array([100.0, -1.0]), "Thrust_power")
    fit_and_plot(np.array(data[['Terameters']]), Y, theta ,"Terameters")
    
    #Multivariable
    X1 = np.array(data[['Age', 'Thrust_power','Terameters']])
    plot_multivarable(X1, Y)
    



