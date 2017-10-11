import tensorflow as tf
import numpy as np
import os.path
import math

class DNN:
    def __init__(self, state_size, action_size):
        self.input_layer = tf.placeholder(tf.float32, shape=(None, state_size))

        hidden1 = tf.layers.dense(inputs=self.input_layer, units=state_size * 2, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=state_size * 2, activation=tf.nn.relu)
        hidden3 = tf.layers.dense(inputs=hidden2, units=state_size * 2, activation=tf.nn.relu)

        self.prediction = tf.layers.dense(inputs=hidden3, units=action_size)
        self.expected = tf.placeholder(tf.float32, shape=(None, action_size))

        self.train_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.expected, self.prediction))
        self.train_step = tf.train.AdagradOptimizer(.1).minimize(self.train_loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver = tf.train.Saver()
        self.path = 'train/train.ckpt'
        if os.path.exists(self.path + '.meta'):
            print('loading from ' + self.path)
            self.saver.restore(self.sess, self.path)

    def train(self, X, Y):
        feed_dict = {self.input_layer: X, self.expected: Y}
        loss = 1000
        for i in range(1000):
            loss, _ = self.sess.run([self.train_loss, self.train_step], feed_dict=feed_dict)
        return loss

    def run(self, X):
        return self.sess.run(self.prediction, feed_dict={self.input_layer: X})

    def save(self):
        self.saver.save(self.sess, self.path)
