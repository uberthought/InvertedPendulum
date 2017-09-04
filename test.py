from pendulum import Pendulum
from network import DNN

import numpy as np
import pickle
import os.path
import math
import random

def action_to_acceleration(action):
    if action == 0:
        return -10.0
    elif action == 1:
        return 0.0
    elif action == 2:
        return -10.0


dnn = DNN(Pendulum.state_size, Pendulum.action_size)

# initial_theta = math.pi + (random.random() - 0.5) / 5
# initial_theta = math.pi
initial_theta = (random.random() - 0.5) / 50
pendulum = Pendulum(initial_theta)
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

    print('Score ', score, ' action ', action, ' a ', a)
    print(' actions ', actions)

print('average score ', cumulative_score / iterations)
