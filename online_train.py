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
round = 1
iteration = 0
episode = []

while round <= 27:

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

    if terminal:
        # train
        if (len(episodes) > 0):
            training_episodes = [episode, random.choice(episodes)]
            critic_loss = actorCritic.train_critic(training_episodes, True)
            actor_loss = actorCritic.train_actor(training_episodes, True)
            actorCritic.save()
            print('round', round, 'score', score, 'critic loss', critic_loss, 'actor loss', actor_loss)
        else:
            print('round', round, 'score', score)

        # reset
        round += 1
        episodes.append(episode)
        episode = []
        pickle.dump(episodes, open("episodes.p", "wb"))
        pendulum = Pendulum(Pendulum.random_theta())
        iteration = 0

