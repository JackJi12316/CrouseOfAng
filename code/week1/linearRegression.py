import numpy as np
import matplotlib.pylab as plt

# 采用梯度形式 0.001->0.003->0.01->...
global alpha,size
alpha = 0.001   # learning rate
size = 0.000001 # 控制迭代范围


# 读取数据
def getData(file_name):
    Datalines = np.loadtxt(file_name,delimiter=',')
    Datalines = np.array(Datalines)
    return Datalines


# 设为 h(x)=θ0+θ1*x，利用梯度下降法
# 设置cost function为 J(θ) = (1/m)*Σ0->m (h(x)-y)^2
# 即   J(θ) = (1/2m)*Σ0->m (θ0+θ1*x-y)^2
# 对θ0求偏导 (1/2m)*Σ0->m 2(θ0+θ1*x-y) = (1/m)*Σ0->m (θ0+θ1*x-y)
# 对θ1求偏导 (1/2m)*Σ0->m 2x(θ0+θ1*x-y) = (1/m)*Σ0->m x(θ0+θ1*x-y)
def GradientDescent(x,y):
    theta0 = 0
    theta1 = 0
    while True:
        acc_d_theta0 = 0
        acc_d_theta1 = 0
        for i,j in zip(x,y):
            acc_d_theta0 += (theta0+theta1*i-j)
            acc_d_theta1 += i*(theta0+theta1*i-j)
        acc_d_theta0 /= len(x)
        acc_d_theta1 /= len(x)
        tmp0 = theta0 - alpha * acc_d_theta0
        tmp1 = theta1 - alpha * acc_d_theta1
        if abs(tmp1-theta1)<size:
            theta0 = tmp0
            theta1 = tmp1
            return [theta0,theta1]
        theta0 = tmp0
        theta1 = tmp1


# main function
if __name__ == '__main__':
    array = getData("ex1data1.txt")
    x = [] # population of a city
    y = [] # profit
    for i in array:
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x,y,c='red')
    plt.xlabel('population')
    plt.ylabel('profit')
    theta0 = GradientDescent(x,y)[0]
    theta1 = GradientDescent(x,y)[1]
    z = np.arange(0,30,0.1)
    h = theta0+theta1*z
    plt.plot(z,h,c='b')
    plt.show() # 数据可视化



