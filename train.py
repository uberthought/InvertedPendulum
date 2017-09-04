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
            discount_factor = .001
            actions1[0][action] = score + discount_factor * np.max(actions2)

        X = np.concatenate((X, np.reshape(state0, (1, Pendulum.state_size))), axis=0)
        Y = np.concatenate((Y, actions1), axis=0)

    return dnn.train(X, Y)

def action_to_acceleration(action):
    if action == 0:
        return -10.0
    elif action == 1:
        return 0.0
    elif action == 2:
        return -10.0


dnn = DNN(Pendulum.state_size, Pendulum.action_size)

experiences = []
old_experiences = []
if os.path.exists('old_experiences.p'):
    old_experiences = pickle.load(open("old_experiences.p", "rb"))
print('old_experiences ', len(old_experiences))

def add_experience(experience):
    old_experiences.append(experience)
    experiences.append(experience)

# initial_theta = math.pi
initial_theta = (random.random() - 0.5) / 50
pendulum = Pendulum(initial_theta)
round = 0
score = 0

for i in range(10000000):

    state0 = pendulum.state()

    a = 0.0

    if random.randint(0, 2) == 0:
        action = np.random.choice(Pendulum.action_size, 1)
    else:
        actions = dnn.run([state0])
        action = np.argmax(actions)

    a = action_to_acceleration(action)

    # Take the action (aa) and observe the the outcome state (s′s′) and reward (rr).
    pendulum.rk4_step(pendulum.dt, a)

    state1 = pendulum.state()
    terminal = pendulum.terminal()
    score = pendulum.score()

    experience = {'state0': state0, 'action': action, 'state1': state1, 'score': score, 'terminal': terminal}
    add_experience(experience)

    # print('theta ', abs(math.pi - pendulum.x[2]), ' a ', a, ' Score ', score)

    if terminal:
        round += 1

        # train
        loss = train(dnn, experiences)

        # retrain old experiences
        if len(old_experiences) >= 0:
            random_old_experiences = np.random.choice(old_experiences, 2000).tolist()
            old_loss = train(dnn, random_old_experiences)

        print('round ', round, ' loss ', loss, ' old loss ', old_loss, ' score ', score)

        experiences = []

        pickle.dump(old_experiences, open("old_experiences.p", "wb"))
        dnn.save()

        pendulum = Pendulum(initial_theta)
        score = 0
