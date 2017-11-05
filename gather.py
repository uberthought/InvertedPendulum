#!/usr/bin/python3

from pendulum import Pendulum
from network import ActorCritic

import numpy as np
import pickle
import os.path
import random

actorCritic = ActorCritic(Pendulum.state_size, Pendulum.action_size)

experiences = []
if os.path.exists('experiences.p'):
    experiences = pickle.load(open("experiences.p", "rb"))
print('experiences ', len(experiences))

pendulum = Pendulum(Pendulum.random_theta())
round = 0
iteration = 0
action0 = False

while round < 27:

    state0 = pendulum.state()

    actions = actorCritic.run_actor([state0])
    if random.random() < 0.5:
        action1 = np.random.choice(Pendulum.action_size, 1)[0]
    else:
        action1 = np.argmax(actions)


    # Take the action1 (aa) and observe the the outcome state (s′s′) and reward (rr).
    pendulum.rk4_step(pendulum.dt, action1)

    state1 = pendulum.state()
    terminal = pendulum.terminal()
    score1 = pendulum.score()

    if action0:
        experience = {'state0': state0, 'action0': action0, 'state1': state1, 'action1': action1, 'score1': score1}
        experiences.append(experience)
    action0 = action1

    iteration += 1

    if terminal:
        round += 1

        print('round', round, 'score', score1, 'iterations', iteration, 'initial theta', pendulum.initial_theta)

        pickle.dump(experiences, open("experiences.p", "wb"))

        pendulum = Pendulum(Pendulum.random_theta())

        iteration = 0
        action0 = False

