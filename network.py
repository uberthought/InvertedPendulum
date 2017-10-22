import tensorflow as tf
import numpy as np
import os.path
import math

class DNN:
    def __init__(self, state_size, action_size):
        self.state_input = tf.placeholder(tf.float32, shape=(None, state_size))

        with tf.name_scope('actor'):
                actor_hidden1 = tf.layers.dense(inputs=self.state_input, units=16, activation=tf.nn.relu)
                actor_hidden2 = tf.layers.dense(inputs=actor_hidden1, units=16, activation=tf.nn.relu)
                self.actor_prediction = tf.layers.dense(inputs=actor_hidden2, units=action_size)
                self.actor_expected = tf.placeholder(tf.float32, shape=(None, action_size))
                self.actor_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.actor_expected, self.actor_prediction))
                self.actor_train = tf.train.AdagradOptimizer(.1).minimize(self.actor_loss)

        with tf.name_scope('critic'):
                critic_hidden1 = tf.layers.dense(inputs=self.state_input, units=16, activation=tf.nn.relu)
                critic_hidden2 = tf.layers.dense(inputs=critic_hidden1, units=16, activation=tf.nn.relu)
                self.critic_prediction = tf.layers.dense(inputs=critic_hidden2, units=1)
                self.critic_expected = tf.placeholder(tf.float32, shape=(None, 1))
                self.critic_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.critic_expected, self.critic_prediction))
                self.critic_train = tf.train.AdagradOptimizer(.1).minimize(self.critic_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if os.path.exists('train/graph.meta'):
                print("loading training data")
                saver = tf.train.Saver()
                saver.restore(self.sess, 'train/graph')

    def train_actor(self, states, Q):
        feed_dict = {self.state_input: states, self.actor_expected: Q}
        for i in range(1000):
            loss, _ = self.sess.run([self.actor_loss, self.actor_train], feed_dict=feed_dict)
        return loss

    def train_critic(self, states, scores):
        feed_dict = {self.state_input: states, self.critic_expected: scores}
        loss = float('inf')
        i = 0
        while i < 2000 and loss > 0.1:
            i += 1
            loss, _ = self.sess.run([self.critic_loss, self.critic_train], feed_dict=feed_dict)
        return loss

    def actor_run(self, states):
        return self.sess.run(self.actor_prediction, feed_dict={self.state_input: states})

    def critic_run(self, states):
        return self.sess.run(self.critic_prediction, feed_dict={self.state_input: states})

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, 'train/graph')
