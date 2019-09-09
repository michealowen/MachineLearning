import numpy as np
from time import time
'''
实验一
l为初始数组,topK找出前K项,再对前K项sort排序
结论:topK排序,要快于整体排序
'''
l = np.arange(10000000)
np.array(np.random.shuffle(l))

t = time()
ll = l[np.argpartition(l,10)][0:10]
ll = np.sort(ll)
print(ll)
print(time()-t)

t1 = time()
ll = np.sort(l)
print(ll[0:10])
print(time()-t1)

'''
实验二
numpy.sort()的排序方法比较
结论 numpy.sort默认使用quicksort,且速度快于堆排序,和归并排序基本相当
'''

'''
l = np.array([])
l = np.array(np.arange(10000000))
np.random.shuffle(l)
t = time()
np.sort(l)
print(time()-t)
t1 = time()
np.sort(l,0,'mergesort')
print(time()-t1)
'''
