"""
Author: michealowen
Last edited: 2019.12.5,Thursday
种群模块,固定拓扑结构
"""
#encoding=UTF-8

import numpy as np
import genome
import numba
from mnist import MNIST
from activation import sigmoid,new_sigmoid
from normalization import Normalize

#两种变异类型,权值变异和偏置变异
MUTATE_TYPE = ['WEIGHT_MUTATE','BIAS_MUTATE']

class Population:
    """
    种群类,完成个体间的选择,交叉,变异
    """

    def __init__(self,pop_size,generation_size,layers,test_data,mutate_range=2):
        '''
        初始化种群
        Params:
            pop_size:种群大小
            generation_size:遗传代数
            layers:网络的拓扑结构 例如[784,30,10]
            test_data:测试数据
            mute_range:基因变异的幅度,(~5~5)
        '''
        self.pop_size = pop_size
        self.generation_size = generation_size
        self.layers = layers
        self.test_data = test_data
        #self.select_range = select_range
        #self.crossover_rate = crossover_rate
        #self.mutate_rate = 0
        self.mutate_range = mutate_range
        #初始化种群
        self.__get_population()
        self.evolution()
        pass

    def __get_population(self):
        '''
        初始化种群,种群大小为pop_size
        '''
        self.pop = []
        for i in range(self.pop_size):
            self.pop.append(genome.Genome(self.layers))
        pass


    def mutate(self):
        '''
        mutate_rate = e^(fitness/10)-1
        '''
        for p in self.pop:
            #p.gene_string = np.array([s*(np.random.random()*2*self.mutate_range-self.mutate_range) if np.random.random() <= self.mutate_rate
            #else s for s in p.gene_string ])
            p.weights_string = np.array([s*(np.random.random()*2*self.mutate_range-self.mutate_range) if np.random.random() <= self.mutate_rate
            else s for s in p.weights_string ])
            p.bias_string = np.array([s*(np.random.random()*2*self.mutate_range-self.mutate_range) if np.random.random() <= self.mutate_rate
            else s for s in p.bias_string ])
        pass

    def crossover(self):
        '''
        交叉,使用两点交叉,交叉范围与交叉率有关,成正比
        '''
        current_pop = []
        #从种群中两两结合,产生子代
        for i in range(int(self.pop_size/2)):
            #随机取出两个父母基因型
            p1 = self.pop[int(np.random.random() * len(self.pop))]
            self.pop.remove(p1)
            p2 = self.pop[int(np.random.random() * len(self.pop))]
            self.pop.remove(p2)
            #两个基因型交叉,产生两个子代
            #在基因上随机选取一个截取点
            
            '''
            TO DO 改为先交叉神经元(神经元的偏置),再交叉神经元的输入(神经元连接的权重)
            '''
            weights_length = len(p1.weights_string)
            bias_length = len(p1.bias_string)
            
            #index 为交叉点在bias_string中的位置,bias_cross_range为bias_string交叉片段的大小
            index = int(np.random.random()*bias_length)
            bias_cross_range = int(self.crossover_rate*bias_length)

            p1_bias_string = np.array([])
            p1_weights_string = np.array([])
            p2_bias_string = np.array([])
            p2_weights_string = np.array([])           
            p1_bias_string = np.append(p1_bias_string,p1.bias_string[:index])
            p1_bias_string = np.append(p1_bias_string,p2.bias_string[index:index+bias_cross_range])          
            p1_bias_string = np.append(p1_bias_string,p1.bias_string[index+bias_cross_range:])

            weights_from_index = self.get_index_in_weights(index)
            weights_to_index = self.get_index_in_weights(index+bias_cross_range)
            p1_weights_string = np.append(p1_weights_string,p1.weights_string[:weights_from_index])
            p1_weights_string = np.append(p1_weights_string,p2.weights_string[weights_from_index:weights_to_index])          
            p1_weights_string = np.append(p1_weights_string,p1.weights_string[weights_to_index:])
            
            p2_bias_string = np.append(p2_bias_string,p2.bias_string[:index])
            p2_bias_string = np.append(p2_bias_string,p1.bias_string[index:index+bias_cross_range])          
            p2_bias_string = np.append(p2_bias_string,p2.bias_string[index+bias_cross_range:])

            p2_weights_string = np.append(p2_weights_string,p2.weights_string[:weights_from_index])
            p2_weights_string = np.append(p2_weights_string,p1.weights_string[weights_from_index:weights_to_index])          
            p2_weights_string = np.append(p2_weights_string,p2.weights_string[weights_to_index:])
            
            p1 = genome.Genome(self.layers,p1_weights_string,p1_bias_string)
            p2 = genome.Genome(self.layers,p2_weights_string,p2_bias_string)
            current_pop.append(p1)
            current_pop.append(p2)
            '''
            gene_length = len(p1.gene_string)
            index = int(np.random.random() * gene_length)
            p1_gene_string = np.array([])
            p1_gene_string = np.append(p1_gene_string,p1.gene_string[:index])
            p1_gene_string = np.append(p1_gene_string,p2.gene_string[index:index+int(self.crossover_rate*gene_length)])          
            p1_gene_string = np.append(p1_gene_string,p1.gene_string[index+int(self.crossover_rate*gene_length):])
            p1 = genome.Genome(self.layers,p1_gene_string)

            p2_gene_string = np.array([])
            p2_gene_string = np.append(p2_gene_string,p2.gene_string[:index])
            p2_gene_string = np.append(p2_gene_string,p1.gene_string[index:index+int(self.crossover_rate*gene_length)])          
            p2_gene_string = np.append(p2_gene_string,p2.gene_string[index+int(self.crossover_rate*gene_length):])
            p2 = genome.Genome(self.layers,p2_gene_string)
            current_pop.append(p1)
            current_pop.append(p2)
            '''
        self.pop = current_pop
        pass

    def get_index_in_weights(self,bias_index):
        '''
        通过交叉点在bias_string的下标,获得在weights_string中的下标
        '''
        weights_index = 0
        for i in range(len(self.layers[1:])):
            if bias_index >= self.layers[i+1]:
                weights_index += self.layers[i+1]*self.layers[i]
                bias_index -= self.layers[i+1]                 
            else:
                weights_index += (bias_index+1)*self.layers[i]
                break
        return weights_index-1

    def select(self):
        '''
        使用轮盘赌法
        '''
        #首先计算种群的适应度
        fitness = []
        for g in self.pop:
            fitness.append(g.get_fitness(self.test_data))

        #交叉率为 1/40*avg(fitness),计算交叉率
        self.crossover_rate = 1/(40*np.average(fitness)) 
        self.mutate_rate = np.exp(np.average(fitness)/10)-1
        print('最大适应度',np.max(fitness),'交叉率',self.crossover_rate)
        current_pop = []
        fitness_summary = np.sum(fitness)
        fitness /= fitness_summary
        i = 0
        while i < self.pop_size:
            index = int(np.random.random()*self.pop_size)
            if fitness[index] >= np.random.random():
                current_pop.append(self.pop[index])
                i+=1
        self.pop = current_pop
        pass

    def evolution(self):
        '''
        进化过程
        不断重复 select,crossover,mutate
        '''
        it = 0
        while it < self.generation_size:
            self.select()
            self.crossover()
            self.mutate()
            it += 1
        pass

if __name__ == '__main__':
    '''
    载入数据,分为训练集和测试集
    '''
    data = MNIST()
    data.gz = True
    train_images, train_labels = data.load_training()
    test_images, test_labels   = data.load_testing()
    train_data = [np.array(train_images),np.array(train_labels)]
    test_data = [np.array(test_images),np.array(test_labels)]
    p = Population(500,20,np.array([784,30,10]),test_data)