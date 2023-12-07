from scipy.io import loadmat
import numpy as np
import matplotlib.pylab as plt

# 计划设置三层神经网络模型:输入层，隐藏层1，输出层
# 输入层设置400个输入神经元(400个灰度色阶)
# 隐藏层1将设置40个神经元，用于处理20行+20列
# 输出层设置10个输出神经元[0~9]

# logistic function
def sigmoid(z):
    return 1/(1+np.exp(-z))


# 获取数据函数
def getData(path):
    dataSet=loadmat(path)
    oldX=dataSet['X']      # X为5000x400的矩阵，400为20x20的灰阶图像
    X = np.insert(oldX,0,1,axis=1)
    y=dataSet['y']      # y为5000x1的矩阵
    print(X.shape,y.shape)
    return X,y
    # y = y.reshape(y.shape[0])       # 将y由[5000,1]改为[1,5000]

# (5000,401)*(401,40)=(5000,40)
def neuralNetwork(X):
    np.random.seed(0)
    theta1 = np.random.rand(401,100)     # 第一层：隐层神经元+100个人为设置的神经元
    theta2 = np.random.rand(101, 10)     # 第二层：同理，隐层神经元+100个人为设置的神经元
    z1 = np.dot(X,theta1)             # z1(5000,40)
    tmp_a1 = sigmoid(z1)
    a1=np.insert(tmp_a1,0,1,axis=1)     # a1(5000,41)
    z2=np.dot(a1,theta2)
    h=sigmoid(z2)
    print(h.shape)


if __name__=="__main__":
    picture,ans = getData("ex3data1.mat")
    neuralNetwork(picture)