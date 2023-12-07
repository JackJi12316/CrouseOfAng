import numpy as np
import matplotlib.pylab as plt


from week1.linearRegression import getData

# 观察成绩分布，设h(x)=θ0+θ1x1+θ2x2
# 所以最终函数为 g(x)=1/(1+e^-(θ0+θ1x1+θ2x2))
# Cost function J(θ)= -y*log(h(x))-(1-y)*log(1-h(x))

alpha = 0.001   # 学习率
size = 100000  # 控制范围

# sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

# 梯度下降法
def GradientDescent(x1,x2,y):
    theta = [0,0,0]
    for n in range(size):
        partDiff = [0,0,0]
        for i,j,k in zip(x1,x2,y):
            X = [1,i,j]
            for l in range(3):
                partDiff[l] = partDiff [l] + (k-sigmoid(np.dot(X,theta)))*X[l]
        for l in range(3):
            partDiff[l] = -partDiff[l]/len(y)
        for l in range(3):
            theta[l] = theta[l] - alpha*partDiff[l]
        print(theta)
    return theta


if __name__=="__main__":
    array = getData("ex2data1.txt")
    exam1 = []
    exam2 = []
    result = []
    for i in array:
        exam1.append(i[0])
        exam2.append(i[1])
        result.append(i[2])
    for i in range(len(result)):
        if result[i] :
            plt.scatter(exam1[i],exam2[i],c="r",marker='o')
        else:
            plt.scatter(exam1[i],exam2[i],c="b",marker='x')
    theta = GradientDescent(exam1,exam2,result)
    z = np.arange(25, 100, 0.1)
    h = -(theta[0]+theta[1]*z)/theta[2]
    plt.plot(z,h)
    plt.xlabel("exam1")
    plt.ylabel("exam2")
    plt.show()

