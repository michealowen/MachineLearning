"""
Author: michealowen
Last edited: 2019.11.28,Thursday
神经网络模块
"""
#encoding=UTF-8

import numpy as np
from mnist import MNIST
from numba import jit
import math
#import tensorflow as tf

class DRNN:
    '''
    Digital Recognition Neural Network
    '''

    def __init__(self,layers):
        '''
        Parms:
            layers:指明每一层的神经元个数 denote the number of the neurons of per layer,type(layer):ndarray
            b:表示偏置
            w:表示权重
            Z:表示线性函数W.T*X+b
            alpha:表示sigmoid(Z)
        '''
        self.layers = layers
        self.layersNum = len(layers)
        self.bias = np.array([np.random.randn(x,1) for x in layers[1:]])
        self.weights = np.array([np.random.randn(x,y) for x,y in zip(layers[:-1],layers[1:])])   #(784,10)(10,10)
        self.Z = np.array([np.zeros(b.shape) for b in self.bias])
        self.alpha = np.array([np.zeros(b.shape) for b in self.bias])

    def Normalization(self,x):
        '''
        归一化
        '''
        #for i in range(x.shape[1]):
        #    x[:,i] = x[:,i]/51
        x = x/255
        return x

    def exponential_decay(self, p, init=1, m=100, finish=0.1):
        '''
        学习率的指数衰减,p为当前时间,init为初始学习率(默认为1),m为终止时间,finish为终止学习率
        '''
        alpha =  np.log(init / finish) / m
        l     = -np.log(init) / alpha
        decay =  np.exp(-alpha * (p + l))
        return decay

    def sigmoid(self,Z):
        #sigmoid函数
        return 1.0/(1.0+np.exp(-Z))
    
    def new_sigmoid(self,z):
        '''
        改进版的sigmoid函数,避免溢出,分两种情况,输入参数为数字和输入参数为向量
        '''
        if isinstance(z, (int, float)):
            if z >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
                z = 1.0 / (1 + np.exp(-z))
            else:
                z = np.exp(z) / (1 + np.exp(z))
        else:
            for row_index, row_value in enumerate(z):
                for column_index , column_value in enumerate(row_value):
                    if column_value >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
                        z[row_index,column_index] = 1.0 / (1 + np.exp(-column_value))
                    else:
                        z[row_index,column_index] = np.exp(column_value) / (1 + np.exp(column_value))        
        return z

    def sigmoid_derivative(self,Z):
        '''
        sigmoid函数的导数
        '''
        return self.sigmoid(Z)*(1-self.sigmoid(Z))
    
    def feed_forward(self,a):
        '''
        通过网络计算输出结果
        '''
        for w,b in zip(self.weights,self.bias):
            a = self.sigmoid(np.dot(a,w)+b.T)
        return a 

    def SBGD(self,train_data,test_data,epochs,batch_size,eta,decay_frequency=50,cost='Quadratic'):
        '''
        小批量随机梯度下降
        Parms:
            train_data:训练数据
            test_data:测试数据
            epochs:迭代次数
            batch_size:批量大小
            eta:每次梯度下降的步长(可调整)
            decay_frequency:学习率更新频率,每多少个batch更新一次
        '''
        self.cost = cost
        #先对数据数据进行归一化处理
        train_data[0]=self.Normalization(train_data[0])
        test_data[0]=self.Normalization(test_data[0])
        former_precision = 0
        current_precision = 0
        #p为当前时间,学习率的参数
        p = 0
        init = eta
        m = epochs*len(train_data[0])/(batch_size*decay_frequency)
        print(m)

        for i in range(epochs):
            #将数据打乱
            current_state = np.random.get_state()  #当前打乱顺序
            np.random.shuffle(train_data[0])
            np.random.set_state(current_state)
            np.random.shuffle(train_data[1])

            #按照batch_size将数据进行分组
            batch_data = (np.array([train_data[0][0+k*batch_size:(k+1)*batch_size] for k in range(int(len(train_data[0])/batch_size))]),
            np.array([train_data[1][0+k*batch_size:(k+1)*batch_size] for k in range(int(len(train_data[1])/batch_size))]))
            for j in range(len(batch_data[0])):
                #每批次的数据对模型进行梯度下降 
                t = self.update_mini_batch(batch_data[0][j],batch_data[1][j],eta)
                self.bias += t[0]
                self.weights += t[1]
                if test_data and j%decay_frequency == 0:
                    p+=1
                    eta = self.exponential_decay(p,init=init,m=m,finish=0.1)
                    print("epoch",i,"batch ",j,"completed","准确率:",self.evaluate(test_data[0],test_data[1]),'学习率',eta)
        return None

    def update_mini_batch(self,mini_batch_data,mini_batch_label,eta):
        '''
        通过每批次数据的权重和偏置的偏导数的和
        '''
        batch_bias = np.array([np.zeros(b.shape) for b in self.bias])
        batch_weights = np.array([np.zeros(w.shape) for w in self.weights])   #(784,10)(10,10)
        for x,y in zip(mini_batch_data,mini_batch_label):
            #首先通过向前计算出各层的数据 Z和alpha
            x = np.reshape(x,newshape=(len(x),1))
            self.forward_propagate(x)
            t = self.back_propagate(x,y,eta)
            batch_bias += t[0]
            batch_weights += t[1]
        return batch_bias,batch_weights

    def forward_propagate(self,x):
        '''
        向前传播,计算每层的Z和alpha
        '''
        for i in range(self.layersNum-1):
            if i == 0:
                self.Z[i] = np.dot(self.weights[i].T,x)
            else:
                self.Z[i] = np.dot(self.weights[i].T,self.alpha[i-1])
            self.alpha[i] = self.sigmoid(self.Z[i])
        return None

    def back_propagate(self,x,y,eta):
        '''
        反向传播,计算权值和偏置的偏导数,保存在micro_bias和micro_weights中
        使用二次代价函数
            x:训练数据
            y:训练数据的标签
            eta:步长
        '''
        micro_bias = np.array([np.zeros(b.shape) for b in self.bias])
        micro_weights = np.array([np.zeros(w.shape) for w in self.weights])   #(784,10)(10,10)
        #print(micro_bias.shape,micro_weights.shape)
        #先计算最后一层,用temp指代 c(损失)对Z的导数
        if self.cost == 'Quadratic':
            temp = self.Quadratic_cost_derivative(y)
        else:
            temp = self.Cross_Entropy_cost_derivative(y)
        micro_bias[-1] -= eta*temp
        micro_weights[-1] -= eta*np.dot(self.alpha[-2],temp.T)       #weights[-1] = (10,1)*(1,10)=(10,10)

        #从导数第二层开始循环
        for i in range(self.layersNum-2)[::-1]:
            #当进行到第一层时,需要使用输入的x进行运算
            if i == 0:
                temp = np.dot(self.weights[1],temp)*self.sigmoid_derivative(self.Z[0])      #temp=(10,10)*(10,1)=(10,1)
                micro_bias[0] -= eta*temp
                micro_weights[0] -= eta*np.dot(x,temp.T)
            else:
                temp = np.dot(self.weights[i+1],temp)*self.sigmoid_derivative(self.Z[i])
                micro_bias[i] -= eta*temp
                micro_weights[i] -=eta*np.dot(self.alpha[i-1],temp.T)
        return micro_bias,micro_weights

    def Quadratic_cost_derivative(self,y):
        '''
        计算最后一层神经元的偏导数,二次代价函数
        '''   
        return 1/(2*self.layers[-1])*(self.alpha[-1]-self.label2vector(y))*self.sigmoid_derivative(self.Z[-1])    #temp = (10,1)plus(10.1)=(10,1) 

    def Cross_Entropy_cost_derivative(self,y):
        '''
        计算最后一层神经元的偏导数,交叉熵代价函数
        优点:在误差较大时可以有较大的偏导数
        '''
        return 1/self.layers[-1]*(self.alpha[-1]-self.label2vector(y))

    def label2vector(self,label):
        '''
        将标签转换为矩阵
        如 (3,4,1,2) ->
        [[0],[0],[0],[1],[0],[0],[0],[0],[0],[0]]
        '''
        label_vector = np.array([0 for i in range(10)])
        label_vector[label] = 1
        label_vector = np.reshape(label_vector,newshape=(10,1))
        return label_vector
    
    def evaluate(self,x,y):
        '''
        计算测试集中预测正确的样本数
        '''
        #输入的x,通过层层计算得到alpha,取最大的一项与y比较,若相等则正确,不相等则错误
        test_alpha = self.feed_forward(x)
        #获取测试结果
        test_result = np.array([np.argmax(test_alpha[i,:]) for i in range(len(test_alpha))])
        return sum(int(i==j) for (i,j) in zip(test_result,y))/len(y)

    def save_model(self,dstFile1,dstFile2):
        '''
        保存model的weights和bias,dstFile1保存偏置,dstFile2保存权重
        '''
        np.save(dstFile1,self.bias)
        np.save(dstFile2,self.weights)
        return None

if __name__ == '__main__':
    
    #修改输入数据
    mdata = MNIST()
    mdata.gz = True
    train_images, train_labels = mdata.load_training()
    test_images, test_labels   = mdata.load_testing()
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    #print(train_images.shape,train_labels.shape)
    train_data = [train_images,train_labels]
    test_data = [test_images,test_labels]
    model = DRNN(np.array([784,30,10]))
    model.SBGD(train_data,test_data,epochs=5,batch_size=100,eta=2,decay_frequency=50)
    #decay_frequency表示每多少个batch更新一次学习率
    