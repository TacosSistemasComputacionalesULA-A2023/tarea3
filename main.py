import os

import gym
import gym_environments
import time
from agent import ValueIteration
import numpy as np

# Allowing environment to have sounds
if "SDL_AUDIODRIVER" in os.environ:
    del os.environ["SDL_AUDIODRIVER"]

# RobotBattery-v0, FrozenLake-v1, FrozenLake-v2
env = gym.make('FrozenLake-v2')
space_num = env.observation_space.n
agent = ValueIteration(space_num, env.action_space.n, env.P, 0.1)

agent.solve(10000, 10000, 0.1, 'valiter', np.zeros(space_num), np.zeros(space_num))
agent.render()
observation, info = env.reset()
terminated, truncated = False, False

env.render()
#time.sleep(2)

while not (terminated or truncated):
    action = agent.get_action(observation)
    observation, _, terminated, truncated, _ = env.step(action)

#time.sleep(2)
env.close()
