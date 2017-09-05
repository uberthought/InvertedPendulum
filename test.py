from pendulum import Pendulum
from network import DNN

import numpy as np
import pickle
import os.path
import math
import random


dnn = DNN(Pendulum.state_size, Pendulum.action_size)

# initial_theta = math.pi + (random.random() - 0.5) / 5
# initial_theta = math.pi
initial_theta = (random.random() - 0.5) / 50
# initial_theta = 0.001
# initial_theta = 0.0
pendulum = Pendulum(initial_theta)
# cumulative_score = 0
iterations = 0
runs = 0

for i in range(10):
    while not pendulum.terminal():

        state0 = pendulum.state()

        actions = dnn.run([state0])
        action = np.argmax(actions)

        score = pendulum.score()
        # print()
        # print('Theta ', pendulum.x[2], ' score ', score, ' a ', Pendulum.action_to_acceleration(action))

        pendulum.rk4_step(pendulum.dt, action)

        terminal = pendulum.terminal()
        score = pendulum.score()

        # cumulative_score += score
        iterations += 1

        # print('Theta ', pendulum.x[2], ' score ', score)
        # print(actions)

    print('iterations ', iterations)
    iterations = 0

    initial_theta = (random.random() - 0.5) / 50
    pendulum = Pendulum(initial_theta)

# print('average score ', cumulative_score / iterations)
