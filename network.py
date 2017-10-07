import tensorflow as tf
import numpy as np
import os.path
import math

class DNN:

    def __init__(self, state_size, action_size):
        self.sess = tf.Session()
        
        self.input_layer = tf.placeholder(tf.float32, shape=(None, 2), name='input')

        saver = tf.train.import_meta_graph('preprocessor/graph.meta', input_map={'preprocessor/input_layer:0': self.input_layer})

        output_layer = tf.get_collection('output_layer')[0]

        with tf.variable_scope('pendulum'):
            hidden1 = tf.layers.dense(inputs=output_layer, units=8, activation=tf.nn.relu, name='hidden_1')
            hidden2 = tf.layers.dense(inputs=hidden1, units=8, activation=tf.nn.relu, name='hidden_2')
            hidden3 = tf.layers.dense(inputs=hidden2, units=8, activation=tf.nn.relu, name='hidden_3')

            self.prediction = tf.layers.dense(inputs=hidden3, units=action_size, name='prediction')
            self.expected = tf.placeholder(tf.float32, shape=(None, action_size), name='expected')

            self.train_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.expected, self.prediction))
            self.train_step = tf.train.AdagradOptimizer(.1).minimize(self.train_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pendulum'))

        self.sess.run(tf.global_variables_initializer())

        if os.path.exists('pendulum/graph.meta'):
            saver.restore(self.sess, 'pendulum/graph')
        saver.restore(self.sess, 'preprocessor/graph')

    def train(self, X, Y):
        feed_dict = {self.input_layer: X, self.expected: Y}
        loss = 0
        for i in range(1000):
            i += 1
            loss, _ = self.sess.run([self.train_loss, self.train_step], feed_dict=feed_dict)

        return loss

    def run(self, X):
        return self.sess.run(self.prediction, feed_dict={self.input_layer: X})

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, 'pendulum/graph')
