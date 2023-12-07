from scipy.io import loadmat
import numpy as np
import matplotlib.pylab as plt


def getData(path:str) -> []:
    dataSet = loadmat(path)
    X = dataSet['X']
    return X


if __name__=="__main__":
    x = []
    y = []
    for i in getData("ex7data1.mat"):
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x,y,c='blue')
    plt.show()