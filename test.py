#!/usr/bin/python3

from pendulum import Pendulum
from network import ActorCritic

import numpy as np
import pickle
import os.path
import math
import random

def run_test(count, actorCritic):
    pendulum = Pendulum(Pendulum.random_theta())
    cumulative_score = 0
    iterations = 0
    cumulative_iterations = 0
    runs = 0

    for i in range(count):
        cumulative_score_run = 0
        while not pendulum.terminal():

            state0 = pendulum.state()

            actions = actorCritic.run_actor([state0])
            action = np.argmax(actions)

            score = pendulum.score()

            pendulum.rk4_step(pendulum.dt, action)

            state1 = pendulum.state()
            terminal = pendulum.terminal()
            score = pendulum.score()

            # print(action, actions, state1[Pendulum.state_size - 1])

            cumulative_score_run += score
            iterations += 1

        print('score final ', score, ' average ', cumulative_score_run / iterations, ' initial theta ', pendulum.initial_theta, ' iterations ', iterations)
        cumulative_score += score
        cumulative_iterations += iterations
        iterations = 0

        pendulum = Pendulum(Pendulum.random_theta())

    return cumulative_score / count, cumulative_iterations / count

actorCritic = ActorCritic(Pendulum.state_size, Pendulum.action_size)
score, iterations = run_test(27, actorCritic)

print('score', score, 'iterations', iterations)

