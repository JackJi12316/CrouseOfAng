import random

from scipy.io import loadmat
import numpy as np
import matplotlib.pylab as plt

color = ["red","blue","green"]


def getData(path:str):
    dataSet = loadmat(path)
    X = dataSet['X']
    x = []
    y = []
    for i in X:
        x.append(i[0])
        y.append(i[1])
    return x,y

def initClusterCentroid(length:int,n:int) -> list[int]:
    randomSub = []
    for count in range(n):
        randomSub.append(random.randint(0,length-1))
    return randomSub

def checkCount(count):
    for c in count:
        if c == 0:
            return False
    return True

def showAllData(x,y):
    plt.scatter(x, y, c='black', marker="*")

def plotListC(c:list)->None:
    for l in range(len(c)):
        plt.plot(c[l][0],c[l][1],c=color[l],marker="x",ms=16)

def plotData(x,y,mark)->None:
    for i in range(len(mark)):
        plt.plot(x[i],y[i],color=color[mark[i]],marker="*",ms=5)

def distance(point1,point2):
    if len(point1) != len(point2):
        raise Exception("Wrong Input!")
    dis = 0
    for l in range(len(point1)):
        dis += np.power(point1[l]-point2[l],2)
    dis = np.sqrt(dis)
    return dis

def computeDistanceToFindCentroid(point:list,centroid:list) -> int:
    distanceList = []
    for i in centroid:
        distanceList.append(distance(point,i))
    sub = 0
    minDistance = distanceList[0]
    i = 1
    while i < len(distanceList):
        if minDistance>distanceList[i]:
            minDistance = distanceList[i]
            sub = i
        i += 1
    return sub

def computeMU(x,y,mark,count,n):
    mu_x = [0]*n
    mu_y = [0]*n
    for l in range(len(mark)):
        mu_x[mark[l]] += x[l]
        mu_y[mark[l]] += y[l]
    for l in range(n):
        mu_x[l] /= count[l]
        mu_y[l] /= count[l]
    return mu_x,mu_y

def assignCentroid(mu_x,mu_y,n):
    conter = []
    for i in range(n):
        conter.append([mu_x[i],mu_y[i]])
    return conter

def drawData(data,centroid):
    mark = [0] * len(data[0])
    for d in range(len(data[0])):
        mark[d] = computeDistanceToFindCentroid([data[0][d], data[1][d]], centroid)
    plotData(data[0], data[1], mark)
    plotListC(centroid)
    plt.show()
    plt.clf()

def KMeans(addr:str,n,flag):
    c = []
    alpha = 0.0001
    x,y = getData(addr)
    mark = [0]*len(x)
    Count = [0]*n

    initSub = initClusterCentroid(len(x),n)
    for count in range(n):
        c.append([x[initSub[count]],y[initSub[count]]])
    if flag:
        plt.scatter(x, y, c='black', marker="*")
        plotListC(c)
        plt.show()
        plt.clf()

    while True:
        for d in range(len(x)):
            mark[d] = computeDistanceToFindCentroid([x[d],y[d]],c)
            Count[mark[d]] += 1
        if not checkCount(Count):
            return KMeans(addr,n,flag)
        if flag:
            plotData(x,y,mark)
            plotListC(c)
            plt.show()
            plt.clf()
        mu_x,mu_y = computeMU(x,y,mark,Count,n)
        Count = [0] * n
        if distance(c[0], [mu_x[0], mu_y[0]]) <= alpha and distance(c[1],[mu_x[1],mu_y[1]]) <= alpha and distance(c[2],[mu_x[2],mu_y[2]]) <= alpha:
            c = assignCentroid(mu_x,mu_y,n)
            break
        c = assignCentroid(mu_x,mu_y,n)

    for d in range(len(x)):
        mark[d] = computeDistanceToFindCentroid([x[d], y[d]], c)
    if flag:
        plotData(x, y, mark)
        plotListC(c)
        plt.show()
        plt.clf()

    ans = sorted(c,key=lambda x:x[1])
    print(ans)
    return ans

# 局部最优解应计算cost function来解决
# TODO
def main():
    i = 0
    centroid = []
    final_center = []
    flag = False
    while i<500:
        flag = not bool(i)
        centroid.append(KMeans("ex7data2.mat",3,flag))
        i += 1
    for i in range(3):
        final_x = 0
        final_y = 0
        for c in centroid:
            final_x += c[i][0]
            final_y += c[i][1]
        final_center.append([final_x/len(centroid),final_y/len(centroid)])
    print(final_center)
    x,y = getData("ex7data2.mat")
    drawData([x,y],final_center)


if __name__=="__main__":
    main()