监督学习

一元线性回归

将训练集中的数据拟合至hθ(x)=θ0+θ1x

![](media/b3da3d7e32719c971f081efaa1164a52.png)

![](media/571686041b2971a5fec24ead445cf07a.png)

梯度下降法

![](media/b94ae75b8fbc1f0a8afed14efe62bf22.png)

alpha (learning rate) means the speed of downhill

通过特征缩放，使所有特征大小保持在一个范围内，提高梯度下降的效率

通常缩放公式使用 （多元线性回归）（μ：平均值）

![](media/3a228c5a8053419dbfc48b400d628a48.png)

![](media/3814c18eaa4641aca4300f809bb225ed.png)

![](media/00b75c0e4896c31481a57c53e55445f3.png)

根据代价函数(cost function)的定义，J(θ)最小时，θ的取值即为所求

即cost function取极小值，根据函数定义此时导数值为0

求解代价函数中θ的值，即求解d J(θ)/dθ=0 ==\> θ0,θ1…,θn

![](media/f831f72c92f27410c48a6d7da09a9771.png)

θ=（XTX）-1XTy (推导，假定X可逆)（注：此公式适用于不可逆矩阵，可逆矩阵可直接通过(1)计算）

=X-1(XT)-1XTy

=X-1y （1）

即

Xθ=y(解矩阵方程组)(X为矩阵，由训练集组成)(θ为待求向量）

![](media/247164e53517c41794af03401fb469ff.png)

善用矩阵

![](media/87e048d9d08f82240234ffec4068e482.png)

Logistic Regression(classitication)

Sigmoid函数:

令hθ(X)=g(θTx)=

![](media/27846cba08a4a6aa3661a35f9ff30759.png)

条件概率：设A、B是两个事件，且P(A)\>0，则称P(B\|A)=为事件A发生的条件下，事件B的条件概率。

hθ(x)=P(y=1\|x;θ)在肿瘤的实例中可以解释为假设肿瘤为恶性肿瘤，则在当前x(θ)的条件下成立的概率是多少

Logistic regression cost function

Note: y=0 or 1 always

写作：

When y=0:*  
*

When y=1:

最终形式：

\*来自统计学中极大自然估计的方法

利用梯度下降法：

![](media/992463652469ae30a21c4632361c84cc.png)

![](media/81d483e3396caebba601425a7893b144.png)

★★★

逻辑回归中多分类问题解决方案：

每次仅关注一个小类，对小类（对此类以外的数据看作负类）进行逻辑回归，最终得到多个逻辑回归模型，作为分类标准

![](media/06f410210ae04605512ad826100645f9.png)

减少欠拟合或过拟合的情况：

![](media/db44c9aa13a92a2ca248b4aadd915c7e.png)

正则化：

如果某些参数值应该较小，则在代价函数中增大参数的代价，使其达到正常值

![](media/6e423b15ff94473246961e6d31db14d3.png)

在代价函数中对每个参数增加惩罚项，使所有参数的值均降低（θ0均可）

λ用于平衡代价函数左侧和右侧

![](media/a2bf67667b95f3dcd155e672f7ce14ce.png)

正规式形式的正则化

![](media/75c689548629ecb4ce8c78444bd215b6.png)

(XTX)-1XT可以求的不可逆矩阵的伪逆矩阵

神经网络

\--非线性分类问题

![](media/83d4157158f7201faadb96e6355da549.png)

输入层-\>隐藏层-\>输出层

![](media/b937e6380f0bcc4176e087374f429a20.png)

![](media/4b730733c3e42797d5e5868e5a55fa99.png)

![](media/54cdbd6987dd26d76a3e94283a151587.png)

L：神经网络中总层数

sl：l层中神经单元的个数（不包括偏置单元）

神经网络代价函数：

![](media/8105f90cbad7879e1c5b06b49fad49ab.png)

注：将所有的 θ均考虑入该表达式

反向传播算法：

![](media/ffa8df4e38333c2f61b77b9de9bc6fc7.png)

即先根据最后一层的结果计算该层的δ

根据已知结果求出上一层δ

最终求出所有层的代价函数

根据代价函数求出最后参数值

δ(l), δ(l-1), δ(l-2),…, δ(2)

Backpropagation algorithm

![](media/4e036787fc420ebf1f740a04de54130e.png)

反向传播算法最终求出的是cost function的偏导数项

神经网络算法：

1 预设θ

2 根据数据集正向计算各级a的值

3 根据算法求出δ(l), δ(l-1), δ(l-2),…, δ(2)

4 根据已有数据求的各级Δ的值

5 根据Δ求出cost function的偏导数项

6 根据求出的偏导数项使用梯度下降法或其他算法求出最佳θ代入神经网络算法中，完成算法训练

![](media/10e2edf49fcec4e513212c4540513768.png)

![](media/a3ebe9f52c70c9d0bdd7bac507b21d35.png)

梯度测试：对前向传播和后向传播的结果进行检测，使梯度下降或其他算法的结果保持正确

![](media/8dcbb4c8ed4d57e53bb3a21c768e0e8d.png)

![](media/045a08f91cebd5c9a53c9c10fc212d95.png)

从数值上估算偏导数的值，与梯度下降或其他算法得到的偏导数值进行比较，进行评估

Check the 算法结果≈估测结果

![](media/e618a9f654db247ef6ca49f8aec831b7.png)

参数初始化时不应使参数相同（均为0），会导致权重经过迭代后依旧相同，出现错误（所有的神经单元均在计算相同的特征值）

使用随机初始化参数的方式

![](media/4a25ddde5ce321bc09cc72895fa8fa06.png)

注：此处ε与梯度检测中的ε无关，为数学概念

![](media/f18d7ca05783546e40ba6e883eaeb368.png)

设计：

1 确定输入单元数

2 确定输出单元数（与最终分类结果有关）

3 隐藏层：默认设置为1；若超过1，则每个隐藏层的节点数目应该相同

算法实现过程：

![](media/b84d0ba1a6149024e52034f605d7f407.png)

![](media/471b19682e312f9942543cc9ba1650ea.png)

评价训练效果

将数据集按7:3进行划分，用70%的数据（训练集training set）进行模型计算，用30%的数据（测试集test set）进行验证

![](media/f54260468be260c13d83acea0b867760.png)

通过对测试集的结果进行度量来评估算法的好坏

步骤：

1 通过training set求出θ-\>hθ(x)

2 通过test set对hθ(x)求误差，进行判断

0/1错误分类法：

![](media/6559425656c685349a5273008c311442.png)

Data set = (training set, validation set, test set) (6:2:2)

![](media/9d525254f6779f1624f42e35f7fce007.png)

以上方法仅能证明对test set的拟合程度好，无法证明对未知的数据可以更好的拟合

因此，将data set设置为三部分

![](media/776fc7b13c8be48d7fccb2797f098f1e.png)

![](media/ea84d06388be3698ad860a6184864659.png)

用数据集进行参数训练，用cross validation set进行参数评估进而选择模型，用test set设置泛化误差作最终结果

欠拟合/过拟合 判断方法—偏差/方差

![](media/a59d4ba2e40fe364d5ab831627c05e79.png)

根据对训练集的cost function与交叉验证集的cost function进行验证得到underfit(high bias)/overfit(high variance)

关于λ的选择：

![](media/8ab175ec21294ce66f49397e44bbf28e.png)

λ大小选择所产生的影响：

![](media/2fc3a7131744d92f8a403653aebcc869.png)

Learning curves：用于检查算法是否存在偏差/方差问题

![](media/9ec71767d22020587d1a87db78a7f52d.png)

高偏差情况：Jcv≈Jtrain is so high

![](media/e72436b0b8b3150a1b0fc5a44d8cdea1.png)

高方差情况：Jcv-Jtrain is petty high(limit m)

![](media/7b83e2996a24b2dea33ca71e657408b2.png)

高方差情况下增大训练集可以提高算法的准确度

![](media/9df83ee6eb9a47078d971d4254ecd434.png)

大型神经网络的潜在问题是过拟合问题，可以通过操作λ解决相关问题

误差分析

算法实现过程：

1 实现一个简单的算法，效果高低不重要

2 根据实现的算法的情况对算法进行改进，可以通过算计计算的方法进行判断是否进行方法的改进（误差分析），也可以通过绘制学习曲线的方式进行修正

Skew classes

Question：如果患癌症的人仅有0.5%，那么使用非机器学习算法y=0的错误率仅有0.5%，甚至优于机器学习算法。

Answer：引入两个参数Precision（查准率）和Recall（召回率）用于评估算法效果

|                 | Actual class |                |                |
|-----------------|--------------|----------------|----------------|
| Predicted class |              | 1              | 0              |
|                 | 1            | True positive  | False positive |
|                 | 0            | False negative | True negative  |

Precision（预测成功/预测总数）（预测成功准确率）（越高越好）

=

预测为1，预测样本确实为1占比

Recall（预测成功/实际人数）

=

预测为1，实际确实为1占比

对Precision(P) and Recall(R) 进行评估的方法

F Score: 2\*

结合平均值和权重的公式（choose the higher lever）

P=0 or R=0 F.score = 0

P=1 and R=1 F.score = 1

通过增加参数数量使算法拥有较低的偏差

通过使用大的训练集使算法拥有较低的方差

多参数+大数据集≈好算法
