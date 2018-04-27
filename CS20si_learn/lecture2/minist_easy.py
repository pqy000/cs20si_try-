import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST", one_hot = True)

x = tf.placeholder(tf.float32,[None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_predicted = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_predicted * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_step.run({x : batch_xs, y_predicted : batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_predicted, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy)
#print(accuracy.eval({x: mnist.test.images, y_predicted: mnist.test.labels}))
