from pendulum import Pendulum
from network import DNN

import numpy as np
import pickle
import os.path
import math
import random

dnn = DNN(Pendulum.state_size, Pendulum.action_size)

pendulum = Pendulum(Pendulum.random_theta())
cumulative_score = 0
iterations = 0
runs = 0
count = 50

for i in range(count):
    cumulative_score_run = 0
    while not pendulum.terminal():

        state0 = pendulum.state()

        actions = dnn.run([state0])
        action = np.argmax(actions)

        score = pendulum.score()

        pendulum.rk4_step(pendulum.dt, action)

        state1 = pendulum.state()
        terminal = pendulum.terminal()
        score = pendulum.score()

        cumulative_score_run += score
        iterations += 1

    print('score final ', score, ' average ', cumulative_score_run / iterations, ' initial theta ', pendulum.initial_theta / (2 * math.pi) * 360 - 180)
    cumulative_score += score
    iterations = 0

    pendulum = Pendulum(Pendulum.random_theta())

print('average final score ', cumulative_score / count)
