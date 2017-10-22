#!/usr/bin/python3

from pendulum import Pendulum
from network import DNN

import numpy as np
import pickle
import os.path
import math
import random

def train_critic(dnn, episode):
    X = np.array([], dtype=np.float).reshape(0, Pendulum.state_size)
    V = np.array([], dtype=np.float).reshape(0, 1)

    cumulative_score = 0

    for experience in reversed(episode):
        state0 = experience['state0']
        action = experience['action']
        state1 = experience['state1']
        score = experience['score']
        terminal = experience['terminal']

        discount_factor = .75
        cumulative_score = score + discount_factor * cumulative_score;

        X = np.concatenate((X, np.reshape(state0, (1, Pendulum.state_size))), axis=0)
        V = np.concatenate((V, [[cumulative_score]]), axis=0)

    return dnn.train_critic(X, V)

def train_actor(dnn, episode):
    X = np.array([], dtype=np.float).reshape(0, Pendulum.state_size)
    Q = np.array([], dtype=np.float).reshape(0, Pendulum.action_size)

    experiences = [i for l in episodes for i in l]

    for experience in experiences:
        state0 = experience['state0']
        action = experience['action']
        state1 = experience['state1']
        score = experience['score']

        actions = dnn.actor_run([state0])

        predicted_score = dnn.critic_run([state1])

        actions[0][action] = score - predicted_score

        X = np.concatenate((X, np.reshape(state0, (1, Pendulum.state_size))), axis=0)
        Q = np.concatenate((Q, actions), axis=0)

    return dnn.train_actor(X, Q)


dnn = DNN(Pendulum.state_size, Pendulum.action_size)

episodes = []
if os.path.exists('episodes.p'):
    episodes = pickle.load(open("episodes.p", "rb"))
print('episodes ', len(episodes))

pendulum = Pendulum(Pendulum.random_theta())
round = 0
score = 1
iteration = 0
cumulative_iterations = 0
episode = []

while round < 27:

    state0 = pendulum.state()

    actions = dnn.actor_run([state0])
    if random.random() < 0.5:
        action = np.random.choice(Pendulum.action_size, 1)[0]
    else:
        action = np.argmax(actions)


    # Take the action (aa) and observe the the outcome state (s′s′) and reward (rr).
    pendulum.rk4_step(pendulum.dt, action)

    state1 = pendulum.state()
    terminal = pendulum.terminal()
    score = pendulum.score()

    experience = {'state0': state0, 'action': action, 'state1': state1, 'score': score, 'terminal': terminal}
    episode.append(experience)

    iteration += 1
    cumulative_iterations += 1

    if terminal:
        round += 1

        # train
        critic_loss = train_critic(dnn, episode)
        actor_loss = train_actor(dnn, episodes)

        average_iterations = cumulative_iterations / round

        print('round ', round, ' critic loss ', critic_loss, ' actor loss ', actor_loss, ' score ', score, ' iterations ', iteration, ' average iterations ', average_iterations, ' initial theta ', pendulum.initial_theta)

        episodes.append(episode)

        episode = []

        pickle.dump(episodes, open("episodes.p", "wb"))
        dnn.save()

        pendulum = Pendulum(Pendulum.random_theta())

        score = 1
        iteration = 0

