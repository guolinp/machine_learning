#!/usr/bin/python

import numpy as np
import tensorflow as tf


learning_rate = 0.08
expected_cost = 0.000000001

# create train data
# y = 0.1*x1 + 0.2*x2 + 0.3
train_x_1 = np.random.random(100)
train_x_2 = np.random.random(100)
# print train_x
arr_y = 0.1 * train_x_1 + 0.2 * train_x_2 + 0.3
# print arr_y

W1 = tf.Variable(tf.random_uniform([1]))
W2 = tf.Variable(tf.random_uniform([1]))
B = tf.Variable(tf.zeros([1]))
Y = W1 * train_x_1 + W2 * train_x_2 + B

cost = tf.reduce_mean(tf.square(Y - arr_y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    sess.run(train)
    if i % 10 == 0:
        curr_cost = sess.run(cost)
        print 'step %5d: cost %.16f' % (i, curr_cost)
        if curr_cost < expected_cost:
            print 'OK.'
            break

curr_w1 = sess.run(W1)  # expected: 0.1
curr_w2 = sess.run(W2)  # expected: 0.2
curr_b = sess.run(B)  # expected: 0.3
print 'Train: W1=%f, W2=%f, B=%f' % (curr_w1, curr_w2, curr_b)

# predict test
x1 = np.array(range(0, 10))  # [0, 1, 2, ..., 9]
x2 = np.array(range(10, 20))  # [10, 11, 12, ..., 19]
y = W1 * x1 + W2 * x2 + B
predict_y = sess.run(y)

for i in range(len(x1)):
    real_y = 0.1 * x1[i] + 0.2 * x2[i] + 0.3
    print 'real     : %.16f' % real_y
    print 'predict  : %.16f' % predict_y[i]
