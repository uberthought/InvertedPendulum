#!/usr/bin/python3

from pendulum import Pendulum
from network import ActorCritic

import pickle
import os.path

actorCritic = ActorCritic(Pendulum.state_size, Pendulum.action_size)

episodes = []
if os.path.exists('episodes.p'):
    episodes = pickle.load(open("episodes.p", "rb"))
print('episodes ', len(episodes))

for i in range(27):
    critic_loss = actorCritic.train_critic(episodes, 4000)
    print('critic loss ', critic_loss)

    actor_loss = actorCritic.train_actor(episodes, 1000)
    print('actor loss ', actor_loss)

    actorCritic.save()
