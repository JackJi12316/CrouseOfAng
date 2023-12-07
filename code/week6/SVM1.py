import random

from scipy.io import loadmat
import matplotlib.pylab as plt
import numpy as np

def initTheta(n)->list:
    theta = []
    for i in range(n):
        theta.append(random.uniform(-1,1))
    return theta

def getData(path:str) -> []:
    dataSet = loadmat(path)
    X = dataSet['X']
    y = dataSet['y']
    return X,y

def cost_0(z):
    if z<=-1:
        return 0
    else:
        return z+1

def cost_1(z):
    if z>=1:
        return 0
    else:
        return 1-z

def costFunction(theta:np.array,X:list,y:list):
    J = 0
    C = 100
    thetaR = np.delete(theta,0,axis=1)
    thetaRT = np.reshape(thetaR.shape[0],1)
    thetaT = np.reshape(theta.shape[0],1)
    for i in range(len(y)):
        J += y[i]*cost_1(np.dot(thetaT,X))+(1-y[i])*cost_0(np.dot(thetaT,X))
    J *= C
    J += np.dot(thetaRT,thetaR)
    return J

def pltData(X,y):
    X_x = []
    X_y = []
    for data in X:
        X_x.append(data[0])
        X_y.append(data[1])
    for i in range(len(y)):
        if y[i]:
            plt.plot(X_x[i], X_y[i], c="green", marker="x")
        else:
            plt.plot(X_x[i], X_y[i], c="red", marker="o")
    plt.show()

def main():
    X,y = getData("ex6data1.mat")
    pltData(X,y)
    x_0 = np.ones(len(X))
    X = np.insert(X,0,x_0,axis=1)
    X = np.array(X)
    y = np.array(y)
    theta = initTheta(len(X[0]))
    theta = np.array(theta)
    thetaT= theta.reshape(theta.shape[0],1)
    print(theta)
    print(thetaT)

if __name__=="__main__":
    main()
