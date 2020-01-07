"""
Author: michealowen
Last edited: 2019.12.14,Saturday
像素基因型模块,确定像素用或不用
"""
#encoding=UTF-8

import numpy as np
#import NeuralNetwork.tf.mnist_train as net

class Genome:
    '''基因型'''

    def __init__(self,propability,gene_length,k):
        '''
        Args:
            propability:每个像素选或者不选的概率
            gene_length:基因的长度
            k:输入数据复杂度的惩罚系数
            '''
        self.propability = propability
        self.gene_length = gene_length
        self.k = k
        pass

    def build_gene(self):
        '''
        随机生成基因型,基因为True的概率为propability
        '''
        self.gene = np.array([ True if np.random.random() <= self.propability else False for i in range(self.gene_length)])
        pass
    
    def get_fitness(self):
        '''
        通过训练模型的准确率和神经网络的输入复杂度计算适应度
        fitness(g) = Accuracy - k*len(True)
        len(True) 表示True的数量
        '''
        #n = net.network([np.sum(self.gene==True),10,10],self.gene)

        pass