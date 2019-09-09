import random
import numpy as np

def makeData():
    f = open('data.txt','r+')
    matrix = [[0,0,0]for i in range(100)]
    i = 0
    while i<100:   
        x = random.randint(-100,100)
        y = random.randint(-100,100)
        if judge([x,y,1])!= 0:
            matrix[i] = [x,y,judge([x,y,1])]
            f.write(str(matrix[i][0])+','+str(matrix[i][1])+','+str(matrix[i][2]))
            if i!= 99:
                f.write('\n')
            i +=1
    f.close()
    return None 

def judge(t):
    [a,b,c] = [2,-1,5]
    if np.dot([a,b,c],t) > 0:
        return 1
    elif np.dot([a,b,c],t) < 0:
        return -1
    else:
        return 0

makeData()    
