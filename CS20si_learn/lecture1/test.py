import tensorflow as tf
import numpy as np
import matplotlib as plt
import xlrd

DATA_FILE = 'data/fire_theft.xls'

book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

X = tf.placeholder(tf.float32, name = "X")
Y = tf.placeholder(tf.float32, name = "Y")

w = tf.Variable(0.0, name = "weights")
b = tf.Variable(0.0, name = "bias")

X1 = np.linspace(-1, 1, 100)
Y1 = X1 * 3 + np.random.randn(X1. shape[0]) * 0.5

Z = np.zeros((100, 2))
Z[:, 0] = X1
Z[:, 1] = Y1

Y_predicted = w * X1 + b
loss = tf.square(Y - Y_predicted, name = "loss")
global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
learning_rate = 0.01 * 0.99 ** tf.cast(global_step, tf.float32)
increment_step = global_step.assign_add(1)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        for x, y in Z:
            sess.run(optimizer, feed_dict={X : x, Y : y})
    w_value, b_value = sess.run([b, w])

    print(w_value, b_value)
