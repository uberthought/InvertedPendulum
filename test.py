from pendulum import Pendulum
from network import DNN
from random import randint

import numpy as np
import pickle
import os.path

def action_to_acceleration(action):
    if action == 0:
        return -10.0
    elif action == 1:
        return -1.0
    elif action == 2:
        return -0.1
    elif action == 3:
        return 0.0
    elif action == 4:
        return 0.1
    elif action == 5:
        return 1.0
    elif action == 6:
        return 10.0


dnn = DNN(Pendulum.state_size, Pendulum.action_size)

pendulum = Pendulum()
cumulative_score = 0
iterations = 0

while not pendulum.terminal():

    state0 = pendulum.state()

    actions = dnn.run([state0])
    action = np.argmax(actions)

    a = action_to_acceleration(action)

    pendulum.rk4_step(pendulum.dt, a)

    terminal = pendulum.terminal()
    score = pendulum.score()

    cumulative_score += score
    iterations += 1

    # print('theta ', pendulum.x[2], ' a ', a, ' Score ', score)

print('average score ', cumulative_score / iterations)
