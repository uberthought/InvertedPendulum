#!/usr/bin/python3

from pendulum import Pendulum
from network import ActorCritic

import pickle
import os.path

actorCritic = ActorCritic(Pendulum.state_size, Pendulum.action_size)

if os.path.exists('experiences.p'):
    experiences = pickle.load(open("experiences.p", "rb"))
print('experiences ', len(experiences))

for i in range(27):
    critic_loss = actorCritic.train_critic(experiences, 4000)
    actor_loss = actorCritic.train_actor(experiences, 4000)

    print('critic', critic_loss, 'actor', actor_loss)

    actorCritic.save()
