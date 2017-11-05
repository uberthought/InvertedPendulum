#!/usr/bin/python3

from pendulum import Pendulum
from network import ActorCritic

import numpy as np
import pickle
import os.path
import math
import random

experiences = []
if os.path.exists('experiences.p'):
    experiences = pickle.load(open("experiences.p", "rb"))
print('experiences ', len(experiences))


def run_test(count, actorCritic):
    pendulum = Pendulum(Pendulum.random_theta())
    cumulative_score = 0
    cumulative_iterations = 0
    runs = 0

    for i in range(count):
        cumulative_score_run = 0
        iterations = 0
        action0 = False
        while not pendulum.terminal():

            state0 = pendulum.state()

            actions = actorCritic.run_actor([state0])
            action1 = np.argmax(actions)

            score = pendulum.score()

            pendulum.rk4_step(pendulum.dt, action1)

            state1 = pendulum.state()
            terminal = pendulum.terminal()
            score1 = pendulum.score()

            if action0:
                experience = {'state0': state0, 'action0': action0, 'state1': state1, 'action1': action1, 'score1': score1}
                experiences.append(experience)
            action0 = action1

            # print(action1, actions, state1[Pendulum.state_size - 1])

            cumulative_score_run += score1
            iterations += 1

        print('score final ', score, ' average ', cumulative_score_run / iterations, ' initial theta ', pendulum.initial_theta, ' iterations ', iterations)
        cumulative_score += score1
        cumulative_iterations += iterations

        pendulum = Pendulum(Pendulum.random_theta())

    return cumulative_score / count, cumulative_iterations / count

actorCritic = ActorCritic(Pendulum.state_size, Pendulum.action_size)
score, iterations = run_test(27, actorCritic)

print('score', score, 'iterations', iterations)

pickle.dump(experiences, open("experiences.p", "wb"))

