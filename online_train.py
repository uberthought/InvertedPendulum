#!/usr/bin/python3

from pendulum import Pendulum
from network import ActorCritic

import numpy as np
import pickle
import os.path
import math
import random

actorCritic = ActorCritic(Pendulum.state_size, Pendulum.action_size)

episodes = []
if os.path.exists('episodes.p'):
    episodes = pickle.load(open("episodes.p", "rb"))
print('episodes ', len(episodes))

pendulum = Pendulum(Pendulum.random_theta())
round = 0
score = 1
iteration = 0
cumulative_iterations = 0
episode = []

while round < 27:

    state0 = pendulum.state()

    actions = actorCritic.run_actor([state0])
    if random.random() < 0.5:
        action = np.random.choice(Pendulum.action_size, 1)[0]
    else:
        action = np.argmax(actions)


    # Take the action (aa) and observe the the outcome state (s′s′) and reward (rr).
    pendulum.rk4_step(pendulum.dt, action)

    state1 = pendulum.state()
    terminal = pendulum.terminal()
    score = pendulum.score()

    experience = {'state0': state0, 'action': action, 'state1': state1, 'score': score, 'terminal': terminal}
    episode.append(experience)

    iteration += 1
    cumulative_iterations += 1

    if terminal:
        round += 1

        # train
        critic_loss = actorCritic.train_critic([episode])
        actor_loss = actorCritic.train_actor(episodes)

        average_iterations = cumulative_iterations / round

        print('round ', round, ' critic loss ', critic_loss, ' actor loss ', actor_loss, ' score ', score, ' iterations ', iteration, ' average iterations ', average_iterations, ' initial theta ', pendulum.initial_theta)

        episodes.append(episode)

        episode = []

        pickle.dump(episodes, open("episodes.p", "wb"))
        actorCritic.save()

        pendulum = Pendulum(Pendulum.random_theta())

        score = 1
        iteration = 0

