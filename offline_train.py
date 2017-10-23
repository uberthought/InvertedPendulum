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


for i in range(27):
    batch = random.sample(episodes, 27)
    critic_loss = actorCritic.train_critic(batch, False)
    actor_loss = actorCritic.train_actor(batch, False)
    actorCritic.save()

    print('critic loss ', critic_loss, 'actor loss ', actor_loss)