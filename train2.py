#!/usr/bin/python3

from pendulum import Pendulum
from network import DNN

import numpy as np
import pickle
import os.path
import math
import random

dnn = DNN(Pendulum.state_size, Pendulum.action_size)

episodes = []
if os.path.exists('episodes.p'):
    episodes = pickle.load(open("episodes.p", "rb"))
print('episodes ', len(episodes))
experiences = [i for l in episodes for i in l]

for i in range(10):
    train_experiences = np.random.choice(experiences, 100).tolist()

    X = np.array([], dtype=np.float).reshape(0, Pendulum.state_size)
    Q = np.array([], dtype=np.float).reshape(0, Pendulum.action_size)

    for experience in train_experiences:
        state0 = experience['state0']
        action = experience['action']
        state1 = experience['state1']
        score = experience['score']

        actions = dnn.actor_run([state0])

        predicted_score = dnn.critic_run([state1])

        actions[0][action] = score - predicted_score

        X = np.concatenate((X, np.reshape(state0, (1, Pendulum.state_size))), axis=0)
        Q = np.concatenate((Q, actions), axis=0)

    loss = dnn.train_actor(X, Q)

    print(i, loss)

