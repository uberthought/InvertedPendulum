import tensorflow as tf
import numpy as np
import os.path
import math

class DNN:
    def __init__(self, state_size, action_size):
        self.stddev = tf.placeholder_with_default(0.0, [])

        self.input_layer = tf.placeholder(tf.float32, shape=(None, state_size))
        noise_vector = tf.random_normal(shape=tf.shape(self.input_layer), mean=0.0, stddev=self.stddev, dtype=tf.float32)
        self.noise = tf.add(self.input_layer, noise_vector)

        self.hidden1 = tf.layers.dense(inputs=self.noise, units=state_size, activation=tf.nn.relu)
        self.hidden2 = tf.layers.dense(inputs=self.hidden1, units=state_size, activation=tf.nn.relu)
        self.hidden3 = tf.layers.dense(inputs=self.hidden2, units=state_size, activation=tf.nn.relu)
        self.hidden4 = tf.layers.dense(inputs=self.hidden3, units=state_size, activation=tf.nn.relu)

        self.prediction = tf.layers.dense(inputs=self.hidden4, units=action_size)

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
        feed_dict = {self.input_layer: X, self.expected: Y, self.stddev: 0.0}
        # feed_dict = {self.input_layer: X, self.expected: Y}
        loss = 1000
        i = 0
        while i < 100:
        # while i < 500 and loss > 0.001:
            i += 1
            loss, _ = self.sess.run([self.train_loss, self.train_step], feed_dict=feed_dict)

        return loss

    def run(self, X):
        return self.sess.run(self.prediction, feed_dict={self.input_layer: X})

    def save(self):
        self.saver.save(self.sess, self.path)
