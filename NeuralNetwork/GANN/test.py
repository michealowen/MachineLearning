import genome 
from numba import jit
import time
import timeit
import numpy as np
import math

def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result


def get_index_in_weights(bias_index):
    '''
    通过交叉点在bias_string的下标,获得在weights_string中的下标
    '''
    layers = [783,30,10,10]
    weights_index = 0
    for i in range(len(layers[1:])):
        #print(bias_index)
        #print(weights_index)
        if bias_index >= layers[i+1]:
            weights_index += layers[i+1]*layers[i]
            bias_index -= layers[i+1]                 
        else:
            print(bias_index)
            weights_index += (bias_index+1)*layers[i]
            break
    return weights_index-1

@jit(nopython=True)
def Normalization_1():
    '''归一化'''
    x = np.random.randn(100,1000)
    for i in range(100):
        for j in range(100):
            x[i][j] /= 51
    return x

@jit(nopython=True)
def Normalization_2():
    '''归一化'''
    x = np.random.randn(100,1000)
    for i in range(x.shape[1]):
        x[:,i] = x[:,i]/51
    #x = x/51
    return x


@jit								
def foo():
	x = []
	for a in range(10000000):
		x.append(a)

@jit(nopython=True)
def sigmoid():
    #sigmoid函数
    Z=np.random.rand(100,100)
    return np.sum(Z)
    #return 1.0/(1.0+math.exp(-Z))

def foo3():	
	y = []					   
	for b in range(10000000):
		y.append(b)
 
if __name__ == '__main__':

    t1 = timeit.timeit("foo()", setup="from __main__ import foo" ,number=10)
    print(t1)    

    t1 = timeit.timeit("foo()", setup="from __main__ import foo" ,number=10)
    print(t1)

    '''
    t1 = timeit.timeit("foo2()", setup="from __main__ import foo2" ,number=1)
    print(t1)

    t1 = timeit.timeit("foo2()", setup="from __main__ import foo2" ,number=1)
    print(t1)
    '''

    t1 = timeit.timeit("sigmoid()", setup="from __main__ import sigmoid" ,number=10)
    print(t1)