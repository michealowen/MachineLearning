"""
Author: michealowen
Last edited: 2019.12.5,Thursday
固定拓扑结构的神经网络,权值和偏置的基因组
"""
#encoding=UTF-8

import numpy as np
from mnist import MNIST
from activation import sigmoid,new_sigmoid
from normalization import Normalize
from collections import defaultdict

class Genome:
    """
    基因组模块,对网络的权重以及偏置进行混合编码
    权重和偏置使用实数编码
    """

    def __init__(self,layers,gene_string=None):
        '''
        基因的构造函数
        Params:
            layers:各层神经元个数,如[784,30,20,10]
            gene_string:基因的字符串
        '''
        self.layers = layers
        if not isinstance(gene_string,np.ndarray):
            #根据神经网络各层神经元生成基因
            self.__build_gene()
            self.get_string()
        else:
            self.gene_string = gene_string
            self.get_weights_bias()

    def __build_gene(self):
        '''
        根据神经网络各层神经元生成基因
        '''
        '''权重'''
        self.neurons_weights = np.array([np.random.randn(x,y) for x,y in zip(self.layers[:-1],self.layers[1:])])
        #for w in neurons_weights:
        #    print(w.shape)
        '''偏置'''
        self.neurons_bias = np.array([np.random.randn(x,1) for x in self.layers[1:]])
        #for b in neurons_bias:
        #    print(b.shape)
        #self.gene = gene(neurons_weights,neurons_bias)
        return None

    def get_string(self):
        '''
        通过权重和偏置获得基因的字符串表达
        '''
        self.gene_string = np.array([])
        for i in range(len(self.neurons_weights)):
            #将w转置,使得某个神经元的所有输入发生改变
            self.gene_string = np.append(self.gene_string,self.neurons_weights[i].T.reshape(-1))
        for i in range(len(self.neurons_bias)):    
            self.gene_string = np.append(self.gene_string,self.neurons_bias[i].reshape(-1))
        pass

    def get_weights_bias(self):
        '''
        通过基因字符串解析出权重和偏置
        '''
        self.neurons_weights = []
        self.neurons_bias = []
        index = 0
        for (i,j) in zip(self.layers[:-1],self.layers[1:]):
            #self.neurons_weights.append(self.gene_string[index:index+i*j].reshape(i,j))   
            self.neurons_weights.append(self.gene_string[index:index+i*j].reshape(j,i).T)  
            index+=i*j         
        self.neurons_weights = np.array(self.neurons_weights)
        for i in self.layers[1:]:
            self.neurons_bias.append(self.gene_string[index:index+i])
            index+=i
        self.neurons_bias = np.array(self.neurons_bias)
        pass

    def get_fitness(self,test_data,alpha=0.1):
        '''
        获得个体的适应度
        '''
        self.fitness = get_precision(self.neurons_weights,self.neurons_bias,test_data)
        #print(self.fitness)
        return self.fitness

def get_precision(weights,bias,test_data):
    '''
    通过输入的权重和偏置得到准确率,测试集为test_data
    '''
    test_alpha = Normalize(test_data[0])
    for w,b in zip(weights,bias):
        #print(test_alpha.shape,w.shape)
        test_alpha = sigmoid(np.dot(test_alpha,w)+b.T)
    test_result = np.array([np.argmax(test_alpha[i,:]) for i in range(len(test_alpha))])
    #统计预测结果
    return sum(int(i==j) for (i,j) in zip(test_result,test_data[1]))/len(test_data[1])

class gene:
    def __init__(self,neurons_weights=None,neurons_bias=None,gene_string=None):
        '''
        基因分为两部分,每个神经连接的权重,每个神经元的偏置(除去输入层)
        Params:
            neurons_weights:三维矩阵形式,和神经网络的weights形式相同
            neurons_bias:三维矩阵形式,和神经网络的bias形式相同
            gene_string:字符串形式的基因
        '''
        #若gene_string为空,则进行自己组装字符串
        if gene_string == None:
            self.neurons_weights = neurons_weights
            self.neurons_bias = neurons_bias
            #将权重和偏置组成字符串
            self.get_string()
        else:
            self.gene_string = gene_string
    
    def get_string(self):
        '''
        通过权重和偏置获得基因的字符串表达
        '''
        self.gene_string = np.array([])
        for i in range(len(self.neurons_weights)):
            self.gene_string = np.append(self.gene_string,self.neurons_weights[i].reshape(-1))
        for i in range(len(self.neurons_bias)):    
            self.gene_string = np.append(self.gene_string,self.neurons_bias[i].reshape(-1))
        pass

class GenomeError:
    def __init__(self, a):
        print("GENOME ERROR: "+str(a))
        pass

if __name__ == '__main__':
    #g = Genome([784,10,10,10])
    '''
    载入数据,分为训练集和测试集
    '''
    data = MNIST()
    data.gz = True
    train_images, train_labels = data.load_training()
    test_images, test_labels   = data.load_testing()
    train_data = [np.array(train_images),np.array(train_labels)]
    test_data = [np.array(test_images),np.array(test_labels)]
    #print(len(test_data[0]))
    
    '''
    acc = []
    for i in range(100):
        g = Genome([784,30,10])
        g.__build_gene()
        acc.append(g.get_fitness(test_data))
    acc.sort(reverse=True)
    print(acc[:20])

    '''
    g = Genome([3,2,2],np.array([i for i in range(14)]))
    print(g.neurons_weights,g.neurons_bias)