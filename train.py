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
            discount_factor = 1
            actions1[0][action] = score + discount_factor * np.max(actions2)

        # print(actions1)

        X = np.concatenate((X, np.reshape(state0, (1, Pendulum.state_size))), axis=0)
        Y = np.concatenate((Y, actions1), axis=0)

    return dnn.train(X, Y)


dnn = DNN(Pendulum.state_size, Pendulum.action_size)

experiences = []
old_experiences = []
if os.path.exists('old_experiences.p'):
    old_experiences = pickle.load(open("old_experiences.p", "rb"))
print('old_experiences ', len(old_experiences))

failed = []
if os.path.exists('failed.p'):
    failed = pickle.load(open("failed.p", "rb"))
print('failed ', failed)

pendulum = Pendulum(Pendulum.random_theta())
round = 0
score = 1

for i in range(10000000):

    state0 = pendulum.state()

    actions = []
    if random.random() < 0.1:
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


    if terminal:
        round += 1

        if score < 0.5:
            failed += [pendulum.initial_theta]
            random.shuffle(failed)

        # add old experiences
        train_experiences = np.random.choice(old_experiences, len(experiences) * 2).tolist()
        train_experiences += experiences

        # train
        loss = train(dnn, train_experiences)

        print('round ', round, ' loss ', loss, ' score ', score, ' failed ', len(failed))

        experiences = []

        pickle.dump(old_experiences, open("old_experiences.p", "wb"))
        pickle.dump(failed, open("failed.p", "wb"))
        dnn.save()

        if len(failed) > 0 and random.random() < 0.5:
            pendulum = Pendulum(failed.pop())
        else:
            pendulum = Pendulum(Pendulum.random_theta())

        score = 1
