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

pendulum = Pendulum(Pendulum.random_theta())
round = 0
score = 1
iteration = 0
cumulative_iterations = 0

for i in range(10000000):

    state0 = pendulum.state()

    actions = []
    if random.random() < 0.25:
        action = np.random.choice(Pendulum.action_size, 1)
        # print('random')
    else:
        actions = dnn.run([state0])
        action = np.argmax(actions)
        # print('dnn')

    # Take the action (aa) and observe the the outcome state (s′s′) and reward (rr).
    pendulum.rk4_step(pendulum.dt, action)

    state1 = pendulum.state()
    terminal = pendulum.terminal()
    score = pendulum.score()

    experience = {'state0': state0, 'action': action, 'state1': state1, 'score': score, 'terminal': terminal}
    experiences.append(experience)
    old_experiences.append(experience)

    # print('score ', score, ' terminal ', terminal)
    # print('theta ', pendulum.x[2], ' a ', Pendulum.action_to_acceleration(action), ' Score ', score)
    # print(actions)
    # print('Theta ', (math.pi - state0[2]) / math.pi, ' score ', score, ' a ', Pendulum.action_to_acceleration(action))
    # print((math.pi - state0[2]) / math.pi, ' ', state0[4], ' ', Pendulum.action_to_acceleration(action))

    iteration += 1
    cumulative_iterations += 1

    if terminal:
        round += 1

        # add old experiences
        train_experiences = np.random.choice(old_experiences, (int)(len(experiences) * 4.0)).tolist()
        train_experiences += experiences

        # train
        loss = train(dnn, train_experiences)

        average_iterations = cumulative_iterations / round

        print('round ', round, ' loss ', loss, ' score ', score, ' iterations ', iteration, ' average iterations ', average_iterations, ' initial theta ', pendulum.initial_theta)

        experiences = []

        if len(old_experiences) > 10000:
            old_experiences = np.random.choice(old_experiences, 10000).tolist()

        pickle.dump(old_experiences, open("old_experiences.p", "wb"))
        dnn.save()

        pendulum = Pendulum(Pendulum.random_theta())
        # print(pendulum.score())

        score = 1
        iteration = 0
