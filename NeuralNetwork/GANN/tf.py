'''
Creat by HuangDandan
dandanhuang@sjtu.edu.cn
2018-08-30

Mnist手写字识别，10分类问题
基于Tensorflow采用2层神经网络预测MNIST数据集手写数字10分类问题:
思路：
先定义好预测值，再定义好损失值，再定义优化求解。现在已经有了一个模型
计算模型的当前的效果和准确度

'''
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
from tensorflow.examples.tutorials.mnist import input_data

#mnist数据输入
'''
在MNIST训练数据集中，mnist.train.images 是一个形状为 [60000, 784] 的张量，
第一个维度数字用来索引图片，第二个维度数字用来索引每张图片中的像素点。
在此张量里的每一个元素，都表示某张图片里的某个像素的强度值，值介于0和1之间。
'''
mnist = input_data.read_data_sets('data/', one_hot=True)

#Network Topologies
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_classes = 10

#Input and outputs
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

#Network Parameters
stddev = 0.1
#权重选择高斯初始化
#关键，out权重矩阵易错
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1],stddev=stddev)),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=stddev))
}

#偏置可以选择0值或者高斯初始化
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
print("Network Ready")

#前向传播，返回10个类别的输出，所有的输出不是神经网络的层，直接返回即可
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['w2']), _biases['b2']))
    return(tf.matmul(layer_2, _weights['out']) + _biases['out'])

#Prediction,定义预测值
pred = multilayer_perceptron(x, weights, biases)

#反向传播
#Loss and optimiser
#损失 交叉熵函数softmax_cross_entropy_with_logits(pred,y) 输入：第一参数：预测值，即一次前向传播的结果；第二参数实际的label值
#平均的loss reduce_mean 表示最后的结果除以n
#tf.nn.softmax_cross_entropy_with_logits()函数参数设置更新！
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#最优化算法：梯度下降求解器
optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

#定义精度值，tf.cast（）将True和False转换为1和0
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, 'float'))

#Initializer
init = tf.global_variables_initializer()
print("Function Ready")

#超参数定义
training_epochs = 20
batch_size = 100
display_step = 4
#Launch the graph
sess = tf.Session()
sess.run(init)
#optimize
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    #Iteraton
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict=feeds)
        avg_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = avg_cost/total_batch
    #display
    if (epoch+1) % display_step == 0:
        print("Epoch:%03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys}
        train_acc = sess.run(accr, feed_dict=feeds)
        print("Train accuracy: %.3f" % (train_acc))
        feeds = {x: mnist.test.images, y: mnist.test.labels}
        test_acc = sess.run(accr, feed_dict=feeds)
        print('Test Accuracy: %.3f' % (test_acc))
print("Optimization Finished")