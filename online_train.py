#!/usr/bin/python3

from pendulum import Pendulum
from network import ActorCritic

import numpy as np
import pickle
import os.path
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
action0 = False

while round < 27:

    state0 = pendulum.state()
    
    actions = actorCritic.run_actor([state0])
    if random.random() < 0.25:
        action1 = np.random.choice(Pendulum.action_size, 1)[0]
    else:
        action1 = np.argmax(actions)

    # Take the action1 (aa) and observe the the outcome state (s′s′) and reward (rr).
    pendulum.rk4_step(pendulum.dt, action1)

    state1 = pendulum.state()
    terminal = pendulum.terminal()
    score1 = pendulum.score()

    # print('action1', action1, 'score1', score1, 'state0', state0)

    if action0:
        experience = {'state0': state0, 'action0': action0, 'state1': state1, 'action1': action1, 'score1': score1, 'terminal': terminal}
        episode.append(experience)
    action0 = action1

    iteration += 1
    cumulative_iterations += 1

    if terminal:
        round += 1
        episodes.append(episode)

        # train
        critic_loss = actorCritic.train_critic(episodes, 4000)
        actor_loss = actorCritic.train_actor(episodes, 500)

        average_iterations = cumulative_iterations / round

        print('round', round, 'critic loss', critic_loss, 'actor loss', actor_loss, 'score1', score1, 'iterations', iteration, 'average iterations', average_iterations, 'initial theta', pendulum.initial_theta)


        episode = []

        pickle.dump(episodes, open("episodes.p", "wb"))
        actorCritic.save()

        pendulum = Pendulum(Pendulum.random_theta())

        score = 1
        iteration = 0
        action0 = False

