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

critic_loss = actorCritic.train_critic(episodes)
actor_loss = actorCritic.train_actor(episodes)
actorCritic.save()

print('critic loss ', critic_loss, 'actor loss ', actor_loss)

