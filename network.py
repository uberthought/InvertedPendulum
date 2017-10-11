import tensorflow as tf
import numpy as np
import os.path
import math

class DNN:
    def __init__(self, state_size, action_size):
        with tf.variable_scope('train'):
                self.input_layer = tf.placeholder(tf.float32, shape=(None, state_size))
                interface_layer = tf.layers.dense(inputs=self.input_layer, units=8, name='interface')

        with tf.variable_scope('hidden'):
                units=8
                hidden_input = tf.placeholder_with_default(input=interface_layer, shape=(None, units))
                hidden1 = tf.layers.dense(hidden_input, units=units, activation=tf.nn.relu)
                hidden2 = tf.layers.dense(inputs=hidden1, units=units, activation=tf.nn.relu)
                hidden3 = tf.layers.dense(inputs=hidden2, units=units, activation=tf.nn.relu)
                hidden_output = tf.layers.dense(inputs=hidden3, units=units)
                tf.add_to_collection('hidden_input', hidden_input)
                tf.add_to_collection('hidden_output', hidden_output)

        # with tf.variable_scope('hidden'):
        #         units=8
        #         hidden_input = tf.placeholder(tf.float32, shape=(None, units))
        #         saver = tf.train.import_meta_graph('hidden/graph.meta', input_map={'hidden/hidden_input:0': hidden_input})
        #         hidden_output = tf.get_collection('hidden_output')[0]

        with tf.variable_scope('train'):
                self.prediction = tf.layers.dense(inputs=hidden_output, units=action_size, name='prediction')
                self.expected = tf.placeholder(tf.float32, shape=(None, action_size), name='expected')

                self.train_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.expected, self.prediction))
                self.train_step = tf.train.AdagradOptimizer(.1).minimize(self.train_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if os.path.exists('train/graph.meta'):
                print("loading training data")
                train_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train'))
                train_saver.restore(self.sess, 'train/graph')

        if os.path.exists('hidden/graph.meta'):
                print("loading hidden data")
                hidden_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden'))
                hidden_saver.restore(self.sess, 'hidden/graph')

        # self.saver = tf.train.Saver()
        # self.path = 'train/train.ckpt'
        # if os.path.exists(self.path + '.meta'):
        #     print('loading from ' + self.path)
        #     self.saver.restore(self.sess, self.path)

    def train(self, X, Y):
        feed_dict = {self.input_layer: X, self.expected: Y}
        loss = 1000
        for i in range(1000):
            loss, _ = self.sess.run([self.train_loss, self.train_step], feed_dict=feed_dict)
        return loss

    def run(self, X):
        return self.sess.run(self.prediction, feed_dict={self.input_layer: X})

    def save(self):
        hidden_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='hidden'))
        hidden_saver.save(self.sess, 'hidden/graph')
        train_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train'))
        train_saver.save(self.sess, 'train/graph')
