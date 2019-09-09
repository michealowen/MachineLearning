"""
Author: Michealowen
Last edited:2019.9.3,Tuesday
SVD奇异矩阵分解
"""
#encoding=UTF-8

import numpy as np
import matplotlib.pyplot as plt

def loadData(imagePath):
    '''
    Args:
        imagePath:图片的路径
    Returns:
        data:图像矩阵
    '''
    image = plt.imread(imagePath)
    #plt.imshow(image)
    #plt.show()
            
    return image

def imageCompress(srcImage,ratio):
    '''
    Args:
        srcImage:处理前的图像
    Returns:
        dstImage:处理后的图像
    '''

    srcImageShape = srcImage.shape
    #result用于显示压缩过的图像
    result = np.zeros(srcImageShape)

    if len(srcImageShape) == 3:
        #若图片为三通道
        #分成三层处理

        u_shape = 0
        s_shape = 0
        vT_shape = 0

        for i in range(3):
            U, sigma, V = svd(srcImage[:,:,i])
            #U, sigma, V = np.linalg.svd(srcImage[:,:,i])
            print(sigma.shape)
            n_sigmas = 0
            temp = 0

            # 计算达到保留率需要的奇异值数量
            while (temp / np.sum(sigma)) < ratio:
                temp += sigma[n_sigmas]
                n_sigmas += 1

            # 构建奇异值矩阵
            S = np.zeros((n_sigmas, n_sigmas))

            for j in range(n_sigmas):
                S[j, j] = sigma[j]

            # 构建结果
            result[:, :, i] = (U[:, 0:n_sigmas].dot(S)).dot(V[0:n_sigmas, :])
            u_shape = U[:, 0:n_sigmas].shape
            s_shape = S.shape
            vT_shape = V[0:n_sigmas, :].shape

        # 归一化到[0, 1]
        for i in range(3):
            MAX = np.max(result[:, :, i])
            MIN = np.min(result[:, :, i])
            result[:, :, i] = (result[:, :, i] - MIN) / (MAX - MIN)

        # 调整到[0, 255]
        result  = np.round(result * 255).astype('int')
    
    # 显示压缩结果
    plt.figure(figsize= (12, 12))
    plt.imshow(result)
    plt.title("Result Image")
    plt.savefig('stadium_compress.jpg')
    plt.show()
    return result




def svd(data):
    '''
    Args:
        data:进行奇异分解的矩阵
    Rerurns:
        U,sigma,V:奇异分解得到的左奇异矩阵,奇异值矩阵,右奇异矩阵
    '''
    train_data = data/1.0
    if train_data.shape[0] <= train_data.shape[1]: 
        # 计算特征值和特征向量
        eval_u,evec_u = np.linalg.eigh(train_data.dot(train_data.T))
        #计算左奇异矩阵
        #降序排列后，逆序输出
        eval_sort_idx_u = np.argsort(eval_u)[::-1]
        # 将特征值对应的特征向量也对应排好序
        eval_u = np.sort(eval_u)[::-1]
        evec_u = evec_u[:,eval_sort_idx_u]
        # 计算奇异值矩阵的逆
        eval_u = np.sqrt(eval_u)
        eval_u_inv = np.linalg.inv(np.diag(eval_u))
        # 计算右奇异矩阵
        evec_part_v = eval_u_inv.dot((evec_u.T).dot(train_data))

        return evec_u, eval_u, evec_part_v
    else:
        # 计算特征值和特征向量
        eval_v,evec_v = np.linalg.eigh(train_data.T.dot(train_data))
        #计算右奇异矩阵
        #降序排列后，逆序输出
        eval_sort_idx_v = np.argsort(eval_v)[::-1]
        # 将特征值对应的特征向量也对应排好序
        eval_v = np.sort(eval_v)[::-1]
        evec_v = evec_v[:,eval_sort_idx_v]
        # 计算奇异值矩阵的逆
        eval_v = np.sqrt(eval_v)
        eval_v_inv = np.linalg.inv(np.diag(eval_v))
        # 计算右奇异矩阵
        evec_part_u = train_data.dot(evec_v.dot(eval_v_inv.T))

        return evec_part_u, eval_v, evec_v

srcImage = loadData("stadium.jpeg")
# 显示原图像
plt.figure(figsize= (12, 12))
plt.title("Origin Image")
plt.imshow(srcImage)
plt.show()
imageCompress(srcImage,0.99)