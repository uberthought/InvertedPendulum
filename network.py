import tensorflow as tf
import numpy as np
import os.path
import math

class ActorCritic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # self.stddev = tf.placeholder_with_default(0.0, [])

        self.state_input = tf.placeholder(tf.float32, shape=(None, state_size))
        self.action_input = tf.placeholder(tf.float32, shape=(None, 1))
        # noise_vector = tf.random_normal(shape=tf.shape(self.state_input), mean=0.0, stddev=self.stddev, dtype=tf.float32)
        # noise = tf.add(self.state_input, noise_vector)

        with tf.name_scope('actor'):
            actor_units = 16
            actor_hidden1 = tf.layers.dense(inputs=self.state_input, units=actor_units, activation=tf.nn.relu)
            actor_hidden2 = tf.layers.dense(inputs=actor_hidden1, units=actor_units, activation=tf.nn.relu)
            actor_hidden3 = tf.layers.dense(inputs=actor_hidden2, units=actor_units, activation=tf.nn.relu)
            self.actor_prediction = tf.layers.dense(inputs=actor_hidden3, units=action_size)
            self.actor_expected = tf.placeholder(tf.float32, shape=(None, action_size))
            self.actor_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.actor_expected, self.actor_prediction))
            self.actor_train = tf.train.AdagradOptimizer(.1).minimize(self.actor_loss)

        with tf.name_scope('critic'):
            critic_units = 16
            critic_merge = tf.concat([self.state_input, self.action_input], axis = 1)
            critic_hidden1 = tf.layers.dense(inputs=critic_merge, units=critic_units, activation=tf.nn.relu)
            critic_hidden2 = tf.layers.dense(inputs=critic_hidden1, units=critic_units, activation=tf.nn.relu)
            critic_hidden3 = tf.layers.dense(inputs=critic_hidden2, units=critic_units, activation=tf.nn.relu)
            self.critic_prediction = tf.layers.dense(inputs=critic_hidden3, units=1)
            self.critic_expected = tf.placeholder(tf.float32, shape=(None, 1))
            self.critic_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.critic_expected, self.critic_prediction))
            self.critic_train = tf.train.AdagradOptimizer(.1).minimize(self.critic_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if os.path.exists('train/graph.meta'):
                print("loading training data")
                saver = tf.train.Saver()
                saver.restore(self.sess, 'train/graph')

    def run_actor(self, states):
        return self.sess.run(self.actor_prediction, feed_dict={self.state_input: states})

    def run_critic(self, states, actions):
        return self.sess.run(self.critic_prediction, feed_dict={self.state_input: states, self.action_input: actions})

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, 'train/graph')

    def train_critic(self, episodes, iterations):
        X = np.array([], dtype=np.float).reshape(0, self.state_input.shape[1])
        U = np.array([], dtype=np.float).reshape(0, 1)
        V = np.array([], dtype=np.float).reshape(0, 1)

        for episode in episodes:
            cumulative_value = 0
            for experience in reversed(episode):
                state0 = experience['state0']
                action = experience['action']
                state1 = experience['state1']
                score = experience['score']
                terminal = experience['terminal']

                discount_factor = .99
                cumulative_value = score + discount_factor * cumulative_value
                cumulative_value_sigmoid = (1 / (1 + math.exp(-cumulative_value)) - 0.5) * 2

                X = np.concatenate((X, np.reshape(state0, (1, self.state_size))), axis=0)
                U = np.concatenate((U, [[action]]), axis=0)
                V = np.concatenate((V, [[cumulative_value_sigmoid]]), axis=0)

        # feed_dict = {self.state_input: X, self.critic_expected: V, self.stddev: 0.001}
        feed_dict = {self.state_input: X, self.critic_expected: V, self.action_input: U}
        for i in range(iterations):
            loss, _ = self.sess.run([self.critic_loss, self.critic_train], feed_dict=feed_dict)
        return loss

    def train_actor(self, episodes, iterations):
        X = np.array([], dtype=np.float).reshape(0, self.state_size)
        Q = np.array([], dtype=np.float).reshape(0, self.action_size)

        experiences = [i for l in episodes for i in l]

        for experience in experiences:
            state0 = experience['state0']
            action = experience['action']
            state1 = experience['state1']
            score = experience['score']

            actions_vector = np.arange(self.action_size).reshape((self.action_size,1))
            state_vector = [state1] * self.action_size
            predicted_values = self.run_critic(state_vector, actions_vector).reshape(self.action_size)

            X = np.concatenate((X, np.reshape(state0, (1, self.state_size))), axis=0)
            Q = np.concatenate((Q, [predicted_values]), axis=0)

        # feed_dict = {self.state_input: X, self.actor_expected: Q, self.stddev: 0.001}
        feed_dict = {self.state_input: X, self.actor_expected: Q}
        for i in range(iterations):
            loss, _ = self.sess.run([self.actor_loss, self.actor_train], feed_dict=feed_dict)
        return loss