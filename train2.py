from pendulum import Pendulum
from network import DNN

import numpy as np
import pickle
import os.path
import math
import random

def train(dnn, experiences):
    X = np.array([], dtype=np.float).reshape(0, Pendulum.state_size)
    Y = np.array([], dtype=np.float).reshape(0, Pendulum.action_size)

    for experience in experiences:
        state0 = experience['state0']
        action = experience['action']
        state1 = experience['state1']
        score = experience['score']
        terminal = experience['terminal']

        actions1 = dnn.run([state0])

        if terminal:
            actions1[0][action] = score
        else:
            actions2 = dnn.run([state1])
            discount_factor = .85
            actions1[0][action] = score + discount_factor * np.max(actions2)

        X = np.concatenate((X, np.reshape(state0, (1, Pendulum.state_size))), axis=0)
        Y = np.concatenate((Y, actions1), axis=0)

    return dnn.train(X, Y)


dnn = DNN(Pendulum.state_size, Pendulum.action_size)

experiences = []
old_experiences = []
if os.path.exists('old_experiences.p'):
    old_experiences = pickle.load(open("old_experiences.p", "rb"))
print('old_experiences ', len(old_experiences))

for i in range(10000000):

    # pick old experiences
    # train_experiences = np.random.choice(old_experiences, 2000).tolist()

    # train
    loss = train(dnn, old_experiences)

    dnn.save()

    print('iteration ', i, ' loss ', loss)
