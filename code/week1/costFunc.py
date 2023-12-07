import numpy as np
import matplotlib.pylab as plt
import week1.linearRegression as lR
from mpl_toolkits.mplot3d import Axes3D

# J(θ)=(1/2m)*Σ(1->m) (h(x)-y)^2

# 采用梯度形式 0.001->0.003->0.01->...
global alpha,size
alpha = 0.001   # learning rate
size = 1000

# 设为 h(x)=θ0+θ1*x，利用梯度下降法
# 设置cost function为 J(θ) = (1/2m)*Σ0->m (h(x)-y)^2
# 即   J(θ) = (1/2m)*Σ0->m (θ0+θ1*x-y)^2
# 对θ0求偏导 (1/2m)*Σ0->m 2(θ0+θ1*x-y) = (1/m)*Σ0->m (θ0+θ1*x-y)
# 对θ1求偏导 (1/2m)*Σ0->m 2x(θ0+θ1*x-y) = (1/m)*Σ0->m x(θ0+θ1*x-y)
if __name__ == '__main__':
    array = lR.getData("ex1data1.txt")
    x = []  # population of a city
    y = []  # profit
    for i in array:
        x.append(i[0])
        y.append(i[1])
    t0 = []
    t1 = []
    theta0 = 0
    theta1 = 0
    number=0
    while True:
        acc_d_theta0 = 0
        acc_d_theta1 = 0
        for i,j in zip(x,y):
            acc_d_theta0 += (theta0+theta1*i-j)
            acc_d_theta1 += i*(theta0+theta1*i-j)
        acc_d_theta0 /= len(x)
        acc_d_theta1 /= len(x)
        theta0 = theta0 - alpha * acc_d_theta0
        theta1 = theta1 - alpha * acc_d_theta1
        t0.append(theta0)
        t1.append(theta1)
        if number>size:
            break
        number=number+1

    J = []
    for i,j in zip(t0,t1):
        accJ=0
        for u,v in zip(x,y):
            accJ+=pow((i+j*u-v),2)
        J.append((1/2*len(x))*accJ)
    plt.plot(t0,J)
    plt.plot(t1,J)
    plt.show()