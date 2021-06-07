import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train_steps =10000
def autoNorm(data):                           #传入一个矩阵
    mins = data.min(0)                        #返回data矩阵中每一列中最小的元素，返回一个列表
    maxs = data.max(0)                        #返回data矩阵中每一列中最大的元素，返回一个列表
    ranges = maxs - mins                     #最大值列表 -最小值列表 =差值列表
    normData = np.zeros(np.shape(data))      #生成一个与 data矩阵同规格的normData全0矩阵，用于装归一化后的数据
    row =data.shape[0]                        ##返回 data矩阵的行数
    normData =data -np.tile(mins,(row,1))      ##data矩阵每一列数据都减去每一列的最小值
    normData =normData / np.tile(ranges,(row,1))   #data矩阵每一列数据都除去每一列的差值（差值 = 某列的最大值- 某列最小值）
    return normData

#读取训练集



with open('F:/motor/振动数据检测/data_packet.csv','r', encoding="utf-8") as file:
    reader = csv.reader(file)
    a = []
    for item in reader:
        a.append(item)
a =[[float(x) for x in item] for item in a]      #将矩阵数据转化为浮点型
data = np.array(a)
x_data = autoNorm(data[:,0:12])

y_data = data[:,[12,13]]
# x_data = x_data.astype(np.float32)
# y_data = y_data.astype(np.float32)
#print(x_data)
x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0,shuffle=True)


w1=tf.Variable(tf.random.normal([12, 16]),dtype = tf.float32)
w2=tf.Variable(tf.random.normal([16, 8]),dtype = tf.float32)
w3=tf.Variable(tf.random.normal([8, 2]),dtype = tf.float32)

with tf.name_scope('input'):
    x_data_ = tf.placeholder(tf.float32, [None, 12],name='x_data_')
    y_data_ = tf.placeholder(tf.float32, [None, 2],name='y_data_')

bias_1 = tf.Variable(tf.zeros([1, 16]), dtype = tf.float32)
bias_2 = tf.Variable(tf.zeros([1, 8]), dtype = tf.float32)
bias_3 = tf.Variable(tf.zeros([1, 2]), dtype = tf.float32)

#定义前向传播过程
y_model_1 = tf.add(tf.matmul(x_data_ , w1), bias_1)
out_put1 = tf.nn.sigmoid(y_model_1)
y_model_2 = tf.add(tf.matmul(out_put1 , w2), bias_2)
out_put2 = tf.nn.relu(y_model_2)
y_model = tf.nn.softmax(tf.add(tf.matmul(out_put2, w3), bias_3,name='prob'))

#定义损失函数和反向传播算法
#loss = -tf.reduce_sum(y_data_ * tf.log(y_model + 1e-10))
loss = tf.reduce_mean(-tf.reduce_sum(y_data_ * tf.log(y_model + 1e-10),1))         #因为我们标签一般是one_hot 编码  用于写损失函数 必须要  tf.reduse_sum   尾号后面是1   按行和
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y_data_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



with tf.Session() as sess:
    sample=[]
    l_loss=[]
    l_ce =[]
    l_yan =[]
    train_max = 0
    yanzhen_max = 0
    sess.run(tf.global_variables_initializer())
    print("Start training!")
    for i in range(30000):

        sample.append(i)
        sess.run(train_step, feed_dict={x_data_:x_data_train, y_data_: y_data_train})

            #print('Loss（train set）:%.2f' % (sess.run(loss, feed_dict={x_data_:x_data_train, y_data_: y_data_train})))
        loss1 = sess.run(loss, feed_dict={x_data_:x_data_train, y_data_: y_data_train})
        l_loss.append(loss1)

    # print('训练集准确率：', sess.run(accuracy, {x_data_:x_data_train, y_data_: y_data_train}))
    # print('测试集准确率：', sess.run(accuracy, {x_data_: x_data_train, y_data_: y_data_train}))
        ce = sess.run(accuracy, {x_data_:x_data_train, y_data_: y_data_train})
        l_ce.append(ce)
        yan = sess.run(accuracy, {x_data_:x_data_test, y_data_: y_data_test})
        l_yan.append(yan)

        if ce > train_max:
            train_max = ce
            train_i = i
            print(" 训练集  当前最优准确率：" + "在第" + str(train_i) + "代" + str(train_max))

        if yan > yanzhen_max:
            yanzhen_max = yan
            yanzhen_i = i
            print(" 测试集  当前最优准确率：" + "在第" + str(yanzhen_i) + "代" + str(yanzhen_max))

    plt.plot(sample, l_loss, marker="*", linewidth=1, linestyle="--", color="red")
    plt.title("The variation of the loss")
    plt.xlabel("Sampling Point")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    plt.plot(sample, l_ce, marker="*", linewidth=1, linestyle="--", color="red")
    plt.plot(sample, l_yan, marker="*", linewidth=1, linestyle="--", color="blue")
    plt.title("The acc of the train set and verification set")
    plt.xlabel("Sampling Point")
    plt.ylabel("acc")
    plt.grid(True)
    plt.show()

