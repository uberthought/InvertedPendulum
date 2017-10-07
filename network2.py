import tensorflow as tf
import numpy as np
import os.path
import math

sess = tf.Session()

with tf.variable_scope('preprocessor'):
    input_layer = tf.placeholder(tf.float32, shape=(None, 2), name='input_layer')
    hidden1 = tf.layers.dense(inputs=input_layer, units=16, activation=tf.nn.tanh, name='hidden_1')
    # hidden2 = tf.layers.dense(inputs=hidden1, units=8, activation=tf.nn.tanh, name='hidden_2')
    # hidden3 = tf.layers.dense(inputs=hidden1, units=8, activation=tf.nn.tanh, name='hidden_3')
    output_layer = tf.layers.dense(inputs=hidden1, units=4, name='output_layer')

    tf.add_to_collection('input_layer', input_layer)
    tf.add_to_collection('output_layer', output_layer)

with tf.variable_scope('preprocessor_training'):
    expected = tf.placeholder(tf.float32, shape=(None, 4), name='expected')

    train_loss = tf.reduce_mean(tf.losses.mean_squared_error(expected, output_layer))
    train_step = tf.train.AdagradOptimizer(.1).minimize(train_loss)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
if os.path.exists('preprocessor/graph.meta'):
    saver.restore(sess, 'preprocessor/graph')

for i in range(100):
    X = np.random.random_sample((1000, 2)) * 2 * math.pi
    Y = [np.sin(foo).tolist() + np.cos(foo).tolist() for foo in X]
    feed_dict = {input_layer: X, expected: Y}
    loss = 1000
    for j in range(100):
        loss, _ = sess.run([train_loss, train_step], feed_dict=feed_dict)
        print(loss)

saver.save(sess, 'preprocessor/graph')

X1 = [[0, math.pi]]
Y1 = [np.sin(foo).tolist() + np.cos(foo).tolist() for foo in X1]
foo3, foo1, foo2 = sess.run([train_loss, output_layer, expected], feed_dict={input_layer: X1, expected: Y1})
print(foo1[0])
print(foo2[0])
print(foo3)
exit()
