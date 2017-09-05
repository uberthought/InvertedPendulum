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
cumulative_iterations = 0
iterations = 0
runs = 0
count = 20

for i in range(count):
    while not pendulum.terminal():

        state0 = pendulum.state()

        actions = dnn.run([state0])
        action = np.argmax(actions)

        score = pendulum.score()
        # print()
        # print('Theta ', (math.pi - state0[2]) / math.pi, ' score ', score, ' a ', Pendulum.action_to_acceleration(action))

        pendulum.rk4_step(pendulum.dt, action)

        state1 = pendulum.state()
        terminal = pendulum.terminal()
        score = pendulum.score()

        iterations += 1

        # print('Theta ', math.pi - state0[2], ' score ', score)
        # print(actions)

    print('iterations ', iterations)
    cumulative_iterations += iterations
    iterations = 0

    initial_theta = (random.random() - 0.5) / 50
    pendulum = Pendulum(initial_theta)

print('average iterations ', cumulative_iterations / count)
